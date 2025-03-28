from chromadb.errors import NotFoundError

from vectorcode.cli_utils import Config
from vectorcode.common import get_client, get_collection


async def drop(config: Config) -> int:
    client = await get_client(config)
    try:
        collection = await get_collection(client, config)
        collection_path = collection.metadata["path"]
        await client.delete_collection(collection.name)
        print(f"Collection for {collection_path} has been deleted.")
        return 0
    except (ValueError, NotFoundError):
        print(f"There's no existing collection for {config.project_root}")
        return 1
