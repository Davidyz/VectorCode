import logging

from chromadb.errors import InvalidCollectionException

from vectorcode.cli_utils import Config
from vectorcode.common import get_client, get_collection

logger = logging.getLogger(name=__name__)


async def drop(config: Config) -> int:
    client = await get_client(config)
    try:
        collection = await get_collection(client, config)
        collection_path = collection.metadata["path"]
        await client.delete_collection(collection.name)
        print(f"Collection for {collection_path} has been deleted.")
        logger.info(f"Deteted collection at {collection_path}.")
        return 0
    except (ValueError, InvalidCollectionException) as e:
        logger.error(
            f"{e.__class__.__name__}: There's no existing collection for {config.project_root}"
        )
        return 1
