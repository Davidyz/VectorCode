import json
import logging

from chromadb.api.types import IncludeEnum

from vectorcode.cli_utils import Config
from vectorcode.common import ClientManager, get_collection

logger = logging.getLogger(name=__name__)


async def ls(configs: Config) -> int:
    async with ClientManager().get_client(configs=configs) as client:
        try:
            collection = await get_collection(client, configs, False)
        except ValueError:
            logger.error(f"There's no existing collection at {configs.project_root}.")
            return 1
        meta = (await collection.get(include=[IncludeEnum.metadatas])).get("metadatas")
        if meta is None:
            logger.warning("Failed to fetch metadatas from the database.")
            return 0
        paths: list[str] = list(set(str(m.get("path")) for m in meta))
        paths.sort()
        if configs.pipe:
            print(json.dumps(list(paths)))
        else:
            for p in paths:
                print(p)
    return 0
