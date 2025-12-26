import json
import logging
import os

import tabulate

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector

logger = logging.getLogger(name=__name__)


async def ls(configs: Config) -> int:
    result = [
        i.to_dict() for i in await get_database_connector(configs).list_collections()
    ]

    if configs.pipe:
        print(json.dumps(result))
    else:
        table = []
        for meta in result:
            project_root = str(meta["project-root"])
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
                    "Number of Embeddings",
                    "Number of Files",
                    "Embedding Function",
                ],
            )
        )
    return 0
