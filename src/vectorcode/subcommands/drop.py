import logging

from vectorcode.cli_utils import Config
from vectorcode.database.chroma0 import ChromaDB0Connector
from vectorcode.database.errors import CollectionNotFoundError

logger = logging.getLogger(name=__name__)


async def drop(config: Config) -> int:
    database = ChromaDB0Connector(config)
    try:
        await database.drop(str(config.project_root))
        if not config.pipe:
            print(f"Collection for {config.project_root} has been deleted.")
        return 0
    except CollectionNotFoundError:
        logger.warning(f"Collection for {config.project_root} doesn't exist.")
        return 1
