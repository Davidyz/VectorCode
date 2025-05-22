import json
import logging
import os
import socket

import tabulate
from chromadb.api import AsyncClientAPI
from chromadb.api.types import IncludeEnum

from vectorcode.cli_utils import Config, cleanup_path
from vectorcode.common import get_client, get_collections

logger = logging.getLogger(name=__name__)


async def get_collection_list(client: AsyncClientAPI) -> list[dict]:
    result = []
    async for collection in get_collections(client):
        meta = collection.metadata
        document_meta = await collection.get(include=[IncludeEnum.metadatas])
        unique_files = set(
            i.get("path") for i in (document_meta["metadatas"] or []) if i is not None
        )
        result.append(
            {
                "project-root": cleanup_path(meta["path"]),
                "user": meta.get("username"),
                "hostname": socket.gethostname(),
                "collection_name": collection.name,
                "size": await collection.count(),
                "embedding_function": meta["embedding_function"],
                "num_files": len(unique_files),
            }
        )
    return result


async def ls(configs: Config) -> int:
    client = await get_client(configs)
    result: list[dict] = await get_collection_list(client)
    logger.info(f"Found the following collections: {result}")

    if configs.pipe:
        print(json.dumps(result))
    else:
        table = []
        for meta in result:
            project_root = meta["project-root"]
            if os.environ.get("HOME"):
                project_root = project_root.replace(os.environ["HOME"], "~")
            row = [
                project_root,
                meta["size"],
                meta["num_files"],
                meta["embedding_function"],
            ]
            table.append(row)
        print(
            tabulate.tabulate(
                table,
                headers=[
                    "Project Root",
                    "Collection Size",
                    "Number of Files",
                    "Embedding Function",
                ],
            )
        )
    return 0
