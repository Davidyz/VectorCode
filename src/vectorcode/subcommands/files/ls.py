import json
import logging

from vectorcode.cli_utils import Config
from vectorcode.common import ClientManager, get_collection, list_collection_files

logger = logging.getLogger(name=__name__)


async def ls(configs: Config) -> int:
    async with ClientManager().get_client(configs=configs) as client:
        try:
            collection = await get_collection(client, configs, False)
        except ValueError:
            logger.error(f"There's no existing collection at {configs.project_root}.")
            return 1
        paths = await list_collection_files(collection)
        if configs.pipe:
            print(json.dumps(list(paths)))
        else:
            for p in paths:
                print(p)
    return 0
