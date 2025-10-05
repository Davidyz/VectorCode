import logging
from typing import Type

from vectorcode.cli_utils import Config
from vectorcode.database.base import DatabaseConnectorBase

logger = logging.getLogger(name=__name__)


def get_database_connector(config: Config) -> DatabaseConnectorBase:
    """
    It's CRUCIAL to keep the `import`s of the database connectors in the branches.
    This allow them to be lazy-imported. This also allow us to keep the main package
    lightweight because we don't have to include dependencies for EVERY database.

    > Raises a `ValueError` in case the database connector is not supported.
    """
    cls: Type[DatabaseConnectorBase] | None = None

    if not config.db_type.endswith("Connector"):
        config.db_type = f"{config.db_type}Connector"
        logger.debug(f"Correcting the name of the db connector to {config.db_type}")

    match config.db_type:
        case "ChromaDB0Connector":
            from vectorcode.database.chroma0 import ChromaDB0Connector

            cls = ChromaDB0Connector
        case "ChromaDBConnector":
            from vectorcode.database.chroma import ChromaDBConnector

            cls = ChromaDBConnector
        case _:
            raise ValueError(f"Unrecognised database type: {config.db_type}")

    return cls.create(config)


__all__ = ["get_database_connector"]
