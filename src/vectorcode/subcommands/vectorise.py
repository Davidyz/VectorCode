import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from asyncio import Lock
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
    expand_globs,
    expand_path,
)
from vectorcode.common import get_client, get_collection, verify_ef

logger = logging.getLogger(name=__name__)


def hash_str(string: str) -> str:
    """Return the sha-256 hash of a string."""
    return hashlib.sha256(string.encode()).hexdigest()


def get_uuid() -> str:
    return uuid.uuid4().hex


async def chunked_add(
    file_path: str,
    collection: AsyncCollection,
    collection_lock: Lock,
    stats: dict[str, int],
    stats_lock: Lock,
    configs: Config,
    max_batch_size: int,
    semaphore: asyncio.Semaphore,
):
    full_path_str = str(expand_path(str(file_path), True))
    async with collection_lock:
        num_existing_chunks = len(
            (
                await collection.get(
                    where={"path": full_path_str},
                    include=[IncludeEnum.metadatas],
                )
            )["ids"]
        )

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
                return
            chunks.append(str(os.path.relpath(full_path_str, configs.project_root)))
            logger.debug(f"Chunked into {len(chunks)} pieces.")
            metas = []
            for chunk in chunks:
                meta: dict[str, str | int] = {"path": full_path_str}
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
                        metadatas=metas,
                    )
    except (UnicodeDecodeError, UnicodeError):  # pragma: nocover
        logger.warning(f"Failed to decode {full_path_str}.")
        return

    if num_existing_chunks:
        async with stats_lock:
            stats["update"] += 1
    else:
        async with stats_lock:
            stats["add"] += 1


def show_stats(configs: Config, stats):
    if configs.pipe:
        print(json.dumps(stats))
    else:
        print(
            tabulate.tabulate(
                [
                    ["Added", "Updated", "Removed"],
                    [stats["add"], stats["update"], stats["removed"]],
                ],
                headers="firstrow",
            )
        )


def exclude_paths_by_spec(paths: Iterable[str], specs: pathspec.PathSpec) -> list[str]:
    """
    Files matched by the specs will be excluded.
    """
    return [path for path in paths if not specs.match_file(path)]


def include_paths_by_spec(paths: Iterable[str], specs: pathspec.PathSpec) -> list[str]:
    """
    Only include paths matched by the specs.
    """
    return [path for path in paths if specs.match_file(path)]


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


async def vectorise(configs: Config) -> int:
    assert configs.project_root is not None
    client = await get_client(configs)
    try:
        collection = await get_collection(client, configs, True)
    except IndexError:
        print("Failed to get/create the collection. Please check your config.")
        return 1
    if not verify_ef(collection, configs):
        return 1

    files = await expand_globs(
        configs.files or load_files_from_include(str(configs.project_root)),
        recursive=configs.recursive,
        include_hidden=configs.include_hidden,
    )

    if not configs.force:
        gitignore_path = os.path.join(str(configs.project_root), ".gitignore")
        specs = [
            gitignore_path,
        ]
        exclude_spec_path = os.path.join(
            configs.project_root, ".vectorcode", "vectorcode.exclude"
        )
        if os.path.isfile(exclude_spec_path):
            specs.append(exclude_spec_path)
        elif os.path.isfile(GLOBAL_EXCLUDE_SPEC):
            specs.append(GLOBAL_EXCLUDE_SPEC)
        for spec_path in specs:
            if os.path.isfile(spec_path):
                logger.info(f"Loading ignore specs from {spec_path}.")
                with open(spec_path) as fin:
                    spec = pathspec.GitIgnoreSpec.from_lines(fin.readlines())
                files = exclude_paths_by_spec((str(i) for i in files), spec)
    else:  # pragma: nocover
        logger.info("Ignoring exclude specs.")

    stats = {"add": 0, "update": 0, "removed": 0}
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

    async with collection_lock:
        all_results = await collection.get(include=[IncludeEnum.metadatas])
        if all_results is not None and all_results.get("metadatas"):
            paths = (meta["path"] for meta in (all_results["metadatas"] or []))
            orphans = set()
            for path in paths:
                if isinstance(path, str) and not os.path.isfile(path):
                    orphans.add(path)
            async with stats_lock:
                stats["removed"] = len(orphans)
            if len(orphans):
                logger.info(f"Removing {len(orphans)} orphaned files from database.")
                await collection.delete(where={"path": {"$in": list(orphans)}})

    show_stats(configs=configs, stats=stats)
    return 0
