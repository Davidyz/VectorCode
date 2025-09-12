import logging

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector
from vectorcode.database.types import ResultType

logger = logging.getLogger(name=__name__)


async def rm(configs: Config) -> int:
    database = get_database_connector(configs)
    remove_count = await database.delete()

    if not configs.pipe:
        print(f"Removed {remove_count} file(s).")
    if await database.count(ResultType.chunk) == 0:
        logger.warning(
            f"The collection at {configs.project_root} is now empty, and will be removed."
        )
        await database.drop()
    return 0
