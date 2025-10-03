import pytest

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector
from vectorcode.database.chroma0 import ChromaDB0Connector


def test_get_database_connector():
    assert isinstance(
        get_database_connector(Config(db_type="ChromaDB0")), ChromaDB0Connector
    )


def test_get_database_connector_invalid_type():
    with pytest.raises(ValueError):
        get_database_connector(Config(db_type="InvalidDB"))
