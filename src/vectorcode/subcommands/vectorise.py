import asyncio
import glob
import hashlib
import json
import logging
import os
import sys
import uuid
from asyncio import Lock
from dataclasses import dataclass, fields
from typing import Iterable, Optional

import pathspec
import tabulate
import tqdm
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import IncludeEnum

from vectorcode.chunking import Chunk, TreeSitterChunker
from vectorcode.cli_utils import (
    GLOBAL_EXCLUDE_SPEC,
    GLOBAL_INCLUDE_SPEC,
    Config,
    SpecResolver,
    expand_globs,
    expand_path,
)
from vectorcode.common import (
    ClientManager,
    get_collection,
    get_embedding_function,
    list_collection_files,
    verify_ef,
)

logger = logging.getLogger(name=__name__)


@dataclass
class VectoriseStats:
    add: int = 0
    update: int = 0
    removed: int = 0
    skipped: int = 0
    failed: int = 0

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, int]:
        return {i.name: getattr(self, i.name) for i in fields(self)}

    def to_table(self) -> str:
        _fields = fields(self)
        return tabulate.tabulate(
            [
                [i.name.capitalize() for i in _fields],
                [getattr(self, i.name) for i in _fields],
            ],
            headers="firstrow",
        )


def hash_str(string: str) -> str:
    """Return the sha-256 hash of a string."""
    return hashlib.sha256(string.encode()).hexdigest()


def hash_file(path: str) -> str:
    """return the sha-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(8192)
            if chunk:
                hasher.update(chunk)
            else:
                break
    return hasher.hexdigest()


def get_uuid() -> str:
    return uuid.uuid4().hex


async def chunked_add(
    file_path: str,
    collection: AsyncCollection,
    collection_lock: Lock,
    stats: VectoriseStats,
    stats_lock: Lock,
    configs: Config,
    max_batch_size: int,
    semaphore: asyncio.Semaphore,
):
    embedding_function = get_embedding_function(configs)
    full_path_str = str(expand_path(str(file_path), True))
    orig_sha256 = None
    new_sha256 = hash_file(full_path_str)
    async with collection_lock:
        existing_chunks = await collection.get(
            where={"path": full_path_str},
            include=[IncludeEnum.metadatas],
        )
        num_existing_chunks = len((existing_chunks)["ids"])
        if existing_chunks["metadatas"]:
            orig_sha256 = existing_chunks["metadatas"][0].get("sha256")
    if orig_sha256 and orig_sha256 == new_sha256:
        logger.debug(
            f"Skipping {full_path_str} because it's unchanged since last vectorisation."
        )
        stats.skipped += 1
        return

    if num_existing_chunks:
        logger.debug(
            "Deleting %s existing chunks for the current file.", num_existing_chunks
        )
        async with collection_lock:
            await collection.delete(where={"path": full_path_str})

    logger.debug(f"Vectorising {file_path}")
    try:
        async with semaphore:
            chunks: list[Chunk | str] = list(
                TreeSitterChunker(configs).chunk(full_path_str)
            )
            if len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == ""):
                # empty file
                logger.debug(f"Skipping {full_path_str} because it's empty.")
                stats.skipped += 1
                return
            chunks.append(str(os.path.relpath(full_path_str, configs.project_root)))
            logger.debug(f"Chunked into {len(chunks)} pieces.")
            metas = []
            for chunk in chunks:
                meta: dict[str, str | int] = {
                    "path": full_path_str,
                    "sha256": new_sha256,
                }
                if isinstance(chunk, Chunk):
                    meta["start"] = chunk.start.row
                    meta["end"] = chunk.end.row

                metas.append(meta)
            async with collection_lock:
                for idx in range(0, len(chunks), max_batch_size):
                    inserted_chunks = chunks[idx : idx + max_batch_size]
                    await collection.add(
                        ids=[get_uuid() for _ in inserted_chunks],
                        documents=[str(i) for i in inserted_chunks],
                        embeddings=embedding_function(
                            list(str(c) for c in inserted_chunks)
                        ),
                        metadatas=metas,
                    )
    except (UnicodeDecodeError, UnicodeError):  # pragma: nocover
        logger.warning(f"Failed to decode {full_path_str}.")
        stats.failed += 1
        return

    if num_existing_chunks:
        async with stats_lock:
            stats.update += 1
    else:
        async with stats_lock:
            stats.add += 1


async def remove_orphanes(
    collection: AsyncCollection,
    collection_lock: Lock,
    stats: VectoriseStats,
    stats_lock: Lock,
):
    async with collection_lock:
        paths = await list_collection_files(collection)
        orphans = set()
        for path in paths:
            if isinstance(path, str) and not os.path.isfile(path):
                orphans.add(path)
        async with stats_lock:
            stats.removed = len(orphans)
        if len(orphans):
            logger.info(f"Removing {len(orphans)} orphaned files from database.")
            await collection.delete(where={"path": {"$in": list(orphans)}})


def show_stats(configs: Config, stats: VectoriseStats):
    if configs.pipe:
        print(stats.to_json())
    else:
        print(stats.to_table())


def exclude_paths_by_spec(
    paths: Iterable[str], spec_path: str, project_root: Optional[str] = None
) -> list[str]:
    """
    Files matched by the specs will be excluded.
    """

    return list(SpecResolver.from_path(spec_path, project_root).match(paths, True))


def load_files_from_include(project_root: str) -> list[str]:
    include_file_path = os.path.join(project_root, ".vectorcode", "vectorcode.include")
    specs: Optional[pathspec.GitIgnoreSpec] = None
    if os.path.isfile(include_file_path):
        logger.debug("Loading from local `vectorcode.include`.")
        with open(include_file_path) as fin:
            specs = pathspec.GitIgnoreSpec.from_lines(
                lines=(os.path.expanduser(i) for i in fin.readlines()),
            )
    elif os.path.isfile(GLOBAL_INCLUDE_SPEC):
        logger.debug("Loading from global `vectorcode.include`.")
        with open(GLOBAL_INCLUDE_SPEC) as fin:
            specs = pathspec.GitIgnoreSpec.from_lines(
                lines=(os.path.expanduser(i) for i in fin.readlines()),
            )
    if specs is not None:
        logger.info("Populating included files from loaded specs.")
        return [
            result.file
            for result in specs.check_tree_files(project_root)
            if result.include
        ]
    return []


def find_exclude_specs(configs: Config) -> list[str]:
    """
    Load a list of paths to exclude specs.
    Can be `.gitignore` or local/global `vectorcode.exclude`
    """
    if configs.recursive:
        specs = glob.glob(
            os.path.join(str(configs.project_root), "**", ".gitignore"), recursive=True
        ) + glob.glob(
            os.path.join(str(configs.project_root), "**", "vectorcode.exclude"),
            recursive=True,
        )
    else:
        specs = [os.path.join(str(configs.project_root), ".gitignore")]

    exclude_spec_path = os.path.join(
        str(configs.project_root), ".vectorcode", "vectorcode.exclude"
    )
    if os.path.isfile(exclude_spec_path):
        specs.append(exclude_spec_path)
    elif os.path.isfile(GLOBAL_EXCLUDE_SPEC):
        specs.append(GLOBAL_EXCLUDE_SPEC)
    specs = [i for i in specs if os.path.isfile(i)]
    logger.debug(f"Loaded exclude specs: {specs}")
    return specs


async def vectorise(configs: Config) -> int:
    assert configs.project_root is not None
    async with ClientManager().get_client(configs) as client:
        try:
            collection = await get_collection(client, configs, True)
        except IndexError as e:
            print(
                f"{e.__class__.__name__}: Failed to get/create the collection. Please check your config."
            )
            return 1
        if not verify_ef(collection, configs):
            return 1

        files = await expand_globs(
            configs.files or load_files_from_include(str(configs.project_root)),
            recursive=configs.recursive,
            include_hidden=configs.include_hidden,
        )

        if not configs.force:
            for spec_path in find_exclude_specs(configs):
                if os.path.isfile(spec_path):
                    logger.info(f"Loading ignore specs from {spec_path}.")
                    files = exclude_paths_by_spec(
                        (str(i) for i in files), spec_path, str(configs.project_root)
                    )
                    logger.debug(f"Files after excluding: {files}")
        else:  # pragma: nocover
            logger.info("Ignoring exclude specs.")

        stats = VectoriseStats()
        collection_lock = Lock()
        stats_lock = Lock()
        max_batch_size = await client.get_max_batch_size()
        semaphore = asyncio.Semaphore(os.cpu_count() or 1)

        with tqdm.tqdm(
            total=len(files), desc="Vectorising files...", disable=configs.pipe
        ) as bar:
            try:
                tasks = [
                    asyncio.create_task(
                        chunked_add(
                            str(file),
                            collection,
                            collection_lock,
                            stats,
                            stats_lock,
                            configs,
                            max_batch_size,
                            semaphore,
                        )
                    )
                    for file in files
                ]
                for task in asyncio.as_completed(tasks):
                    await task
                    bar.update(1)
            except asyncio.CancelledError:
                print("Abort.", file=sys.stderr)
                return 1

        await remove_orphanes(collection, collection_lock, stats, stats_lock)

        show_stats(configs=configs, stats=stats)
        return 0
