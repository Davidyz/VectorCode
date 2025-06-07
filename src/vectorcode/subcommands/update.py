import asyncio
import logging
import os
import sys
from asyncio import Lock

import tqdm
from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException

from vectorcode.cli_utils import Config
from vectorcode.common import get_client, get_collection, verify_ef
from vectorcode.subcommands.vectorise import chunked_add, show_stats

logger = logging.getLogger(name=__name__)


async def update(configs: Config) -> int:
    client = await get_client(configs)
    try:
        collection = await get_collection(client, configs, False)
    except IndexError:
        print("Failed to get/create the collection. Please check your config.")
        return 1
    except (ValueError, InvalidCollectionException):
        print(
            f"There's no existing collection for {configs.project_root}",
            file=sys.stderr,
        )
        return 1
    if collection is None or not verify_ef(collection, configs):
        return 1

    metas = (await collection.get(include=[IncludeEnum.metadatas]))["metadatas"]
    if metas is None:
        return 0
    files_gen = (str(meta.get("path", "")) for meta in metas)
    files = set()
    orphanes = set()
    for file in files_gen:
        if os.path.isfile(file):
            files.add(file)
        else:
            orphanes.add(file)

    stats = {"add": 0, "update": 0, "removed": len(orphanes)}
    collection_lock = Lock()
    stats_lock = Lock()
    max_batch_size = await client.get_max_batch_size()
    semaphore = asyncio.Semaphore(os.cpu_count() or 1)

    with tqdm.tqdm(
        total=len(files), desc="Vectorising files...", disable=configs.pipe
    ) as bar:
        logger.info(f"Updating embeddings for {len(files)} file(s).")
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
        except asyncio.CancelledError:  # pragma: nocover
            print("Abort.", file=sys.stderr)
            return 1

    if len(orphanes):
        logger.info(f"Removing {len(orphanes)} orphaned files from database.")
        await collection.delete(where={"path": {"$in": list(orphanes)}})

    show_stats(configs, stats)
    return 0
