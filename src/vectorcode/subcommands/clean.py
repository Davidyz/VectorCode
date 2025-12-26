import logging

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector

logger = logging.getLogger(name=__name__)


async def clean(configs: Config) -> int:
    database = get_database_connector(configs)
    for removed in await database.cleanup():
        message = f"Deleted collection: {removed}"
        logger.info(message)
        if not configs.pipe:
            print(message)
    return 0
