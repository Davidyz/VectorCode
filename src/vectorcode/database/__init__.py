import logging
from typing import Optional, Type

from vectorcode.cli_utils import Config
from vectorcode.database.base import DatabaseConnectorBase

logger = logging.getLogger(name=__name__)


def get_database_connector(config: Config) -> DatabaseConnectorBase:
    """
    It's CRUCIAL to keep the `import`s of the database connectors in the branches.
    This allow them to be lazy-imported. This also allow us to keep the main package
    lightweight because we don't have to include dependencies for EVERY database.

    """
    cls: Optional[Type[DatabaseConnectorBase]] = None

    match config.db_type:
        case "ChromaDB0Connector":
            from vectorcode.database.chroma0 import ChromaDB0Connector

            cls = ChromaDB0Connector
        case _:
            raise ValueError(f"Unrecognised database type: {config.db_type}")

    return cls.create(config)
