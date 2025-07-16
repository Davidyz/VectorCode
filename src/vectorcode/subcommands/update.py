import asyncio
import logging
import os
import sys
from asyncio import Lock

import tqdm
from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException

from vectorcode.cli_utils import Config
from vectorcode.common import ClientManager, get_collection, verify_ef
from vectorcode.subcommands.vectorise import VectoriseStats, chunked_add, show_stats

logger = logging.getLogger(name=__name__)


async def update(configs: Config) -> int:
    async with ClientManager().get_client(configs) as client:
        try:
            collection = await get_collection(client, configs, False)
        except IndexError as e:
            print(
                f"{e.__class__.__name__}: Failed to get/create the collection. Please check your config."
            )
            return 1
        except (ValueError, InvalidCollectionException) as e:
            print(
                f"{e.__class__.__name__}: There's no existing collection for {configs.project_root}",
                file=sys.stderr,
            )
            return 1
        if collection is None:  # pragma: nocover
            logger.error(
                f"Failed to find a collection at {configs.project_root} from {configs.db_url}"
            )
            return 1
        if not verify_ef(collection, configs):  # pragma: nocover
            return 1

        metas = (await collection.get(include=[IncludeEnum.metadatas]))["metadatas"]
        if metas is None or len(metas) == 0:  # pragma: nocover
            logger.debug("Empty collection.")
            return 0

        files_gen = (str(meta.get("path", "")) for meta in metas)
        files = set()
        orphanes = set()
        for file in files_gen:
            if os.path.isfile(file):
                files.add(file)
            else:
                orphanes.add(file)

        stats = VectoriseStats(removed=len(orphanes))
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
