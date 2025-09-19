import asyncio
import glob
import logging
import os
from asyncio import Lock, Semaphore
from typing import Optional

import pathspec
import tqdm

from vectorcode.cli_utils import (
    GLOBAL_EXCLUDE_SPEC,
    GLOBAL_INCLUDE_SPEC,
    Config,
    SpecResolver,
    expand_globs,
)
from vectorcode.database import get_database_connector
from vectorcode.database.base import DatabaseConnectorBase
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.database.types import ResultType, VectoriseStats
from vectorcode.database.utils import hash_file
from vectorcode.subcommands.vectorise.filter import FilterManager

logger = logging.getLogger(name=__name__)


def show_stats(configs: Config, stats: VectoriseStats):
    if configs.pipe:
        print(stats.to_json())
    else:
        print(stats.to_table())


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


async def vectorise_worker(
    database: DatabaseConnectorBase,
    file_path: str,
    semaphore: Semaphore,
    stats: VectoriseStats,
    stats_lock: Lock,
):
    async with semaphore, stats_lock:
        if os.path.isfile(file_path):
            stats += await database.vectorise(
                file_path=file_path,
            )


async def vectorise(configs: Config) -> int:
    assert configs.project_root is not None
    database = get_database_connector(configs)

    files = await expand_globs(
        configs.files or load_files_from_include(str(configs.project_root)),
        recursive=configs.recursive,
        include_hidden=configs.include_hidden,
    )

    filters = FilterManager()

    try:
        collection_files = (
            await database.list_collection_content(what=ResultType.document)
        ).files

        existing_hashes = set(i.sha256 for i in collection_files)
    except CollectionNotFoundError:
        existing_hashes = set()

    if not configs.force:
        for spec_path in find_exclude_specs(configs):
            # filter by gitignore/vectorcode.exclude
            if os.path.isfile(spec_path):
                logger.info(f"Loading ignore specs from {spec_path}.")
                spec = SpecResolver.from_path(
                    spec_path,
                    str(configs.project_root) if configs.project_root else None,
                )
                filters.add_filter(lambda x: spec.match_file(x, True))

        # filter by sha256
        filters.add_filter(lambda x: hash_file(x) not in existing_hashes)
    else:  # pragma: nocover
        logger.info("Ignoring exclude specs.")

    files = list(filters(files))
    stats = VectoriseStats()
    stats_lock = Lock()
    semaphore = asyncio.Semaphore(os.cpu_count() or 1)

    with tqdm.tqdm(
        total=len(files), desc="Vectorising files...", disable=configs.pipe
    ) as bar:
        try:
            tasks = [
                asyncio.create_task(
                    vectorise_worker(database, file, semaphore, stats, stats_lock)
                )
                for file in files
            ]
            for task in asyncio.as_completed(tasks):
                await task
                bar.update(1)
        except asyncio.CancelledError:  # pragma: nocover
            logger.warning("Abort.")
            return 1

    await database.check_orphanes()

    show_stats(configs=configs, stats=stats)
    return 0
