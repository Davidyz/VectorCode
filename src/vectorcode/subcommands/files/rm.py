import logging
import os
from typing import cast

from chromadb.types import Where

from vectorcode.cli_utils import Config, expand_path
from vectorcode.common import ClientManager, get_collection

logger = logging.getLogger(name=__name__)


async def rm(configs: Config) -> int:
    async with ClientManager().get_client(configs=configs) as client:
        try:
            collection = await get_collection(client, configs, False)
        except ValueError:
            logger.error(f"There's no existing collection at {configs.project_root}.")
            return 1
        paths = list(
            str(expand_path(p, True)) for p in configs.rm_paths if os.path.isfile(p)
        )
        await collection.delete(where=cast(Where, {"path": {"$in": paths}}))
        if not configs.pipe:
            print(f"Removed {len(paths)} file(s).")
        if await collection.count() == 0:
            logger.warning(
                f"The collection at {configs.project_root} is now empty and will be removed."
            )
            await client.delete_collection(collection.name)
    return 0
