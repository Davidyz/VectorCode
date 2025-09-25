import asyncio
import logging
import os
from asyncio import Lock

import tqdm

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector
from vectorcode.database.types import ResultType
from vectorcode.database.utils import hash_file
from vectorcode.subcommands.vectorise import (
    VectoriseStats,
    show_stats,
    vectorise_worker,
)
from vectorcode.subcommands.vectorise.filter import FilterManager

logger = logging.getLogger(name=__name__)


async def update(configs: Config) -> int:
    assert configs.project_root is not None
    database = get_database_connector(configs)

    filters = FilterManager()

    collection_files = (
        await database.list_collection_content(what=ResultType.document)
    ).files

    existing_hashes = set(i.sha256 for i in collection_files)

    files = (i.path for i in collection_files)
    if not configs.force:
        filters.add_filter(lambda x: hash_file(x) not in existing_hashes)
    else:  # pragma: nocover
        logger.info("Ignoring exclude specs.")

    files = list(filters(files))
    stats = VectoriseStats(skipped=len(collection_files) - len(files))
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
        except asyncio.CancelledError:
            logger.warning("Abort.")
            return 1

    await database.check_orphanes()

    show_stats(configs=configs, stats=stats)
    return 0
