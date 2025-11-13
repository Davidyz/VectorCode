import logging

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector
from vectorcode.database.errors import CollectionNotFoundError

logger = logging.getLogger(name=__name__)


async def drop(config: Config) -> int:
    try:
        database = get_database_connector(config)
        await database.drop()
        if not config.pipe:
            print(f"Collection for {config.project_root} has been deleted.")
        return 0
    except CollectionNotFoundError:
        logger.warning(f"Collection for {config.project_root} doesn't exist.")
        return 1
    except Exception:  # pragma: nocover
        raise
