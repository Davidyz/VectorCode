import os

from chromadb.api import AsyncClientAPI

from vectorcode.cli_utils import Config
from vectorcode.common import get_client, get_collections


async def run_clean_on_client(client: AsyncClientAPI, pipe_mode: bool):
    async for collection in get_collections(client):
        meta = collection.metadata
        if await collection.count() == 0 or not os.path.isdir(meta["path"]):
            await client.delete_collection(collection.name)
            if not pipe_mode:
                print(f"Deleted {meta['path']}.")


async def clean(configs: Config) -> int:
    await run_clean_on_client(await get_client(configs), configs.pipe)
    return 0
