import json
import logging

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.database.types import ResultType

logger = logging.getLogger(name=__name__)


async def ls(configs: Config) -> int:
    try:
        database = get_database_connector(configs)
        files = list(
            i.path
            for i in (
                await database.list_collection_content(what=ResultType.document)
            ).files
        )
        if configs.pipe:
            print(json.dumps(files))
        else:
            print("\n".join(files))
        return 0
    except CollectionNotFoundError:
        logger.error(f"There's no existing collection for `{configs.project_root}`.")
        return 1
    except Exception:  # pragma: nocover
        raise
