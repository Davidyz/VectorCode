import json
import logging

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector

logger = logging.getLogger(name=__name__)


async def ls(configs: Config) -> int:
    database = get_database_connector(configs)
    files = list(i.path for i in (await database.list_collection_content()).files)
    if configs.pipe:
        print(json.dumps(files))
    else:
        print("\n".join(files))
    return 0
