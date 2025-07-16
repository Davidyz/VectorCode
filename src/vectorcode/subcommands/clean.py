import logging
import os

from chromadb.api import AsyncClientAPI

from vectorcode.cli_utils import Config
from vectorcode.common import ClientManager, get_collections

logger = logging.getLogger(name=__name__)


async def run_clean_on_client(client: AsyncClientAPI, pipe_mode: bool):
    async for collection in get_collections(client):
        meta = collection.metadata
        logger.debug(f"{meta.get('path')}: {await collection.count()} chunk(s)")
        if await collection.count() == 0 or not os.path.isdir(meta["path"]):
            await client.delete_collection(collection.name)
            logger.info(f"Deleted collection for {meta['path']}")
            if not pipe_mode:
                print(f"Deleted {meta['path']}.")


async def clean(configs: Config) -> int:
    async with ClientManager().get_client(configs) as client:
        await run_clean_on_client(client, configs.pipe)
        return 0
