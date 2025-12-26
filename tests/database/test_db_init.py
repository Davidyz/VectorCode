import sys
from unittest.mock import MagicMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.database import get_database_connector


@pytest.mark.parametrize(
    "db_type, module_to_mock, class_name",
    [
        ("ChromaDB0", "vectorcode.database.chroma0", "ChromaDB0Connector"),
        ("ChromaDB", "vectorcode.database.chroma", "ChromaDBConnector"),
        # To test a new connector, add a tuple here following the same pattern.
        # e.g. ("NewDB", "vectorcode.database.newdb", "NewDBConnector"),
    ],
)
def test_get_database_connector(db_type, module_to_mock, class_name):
    """
    Tests that get_database_connector can correctly return a connector
    for a given db_type. This test is parameterized to be easily
    extensible for new database connectors.
    """
    mock_connector_class = MagicMock()
    mock_module = MagicMock()
    setattr(mock_module, class_name, mock_connector_class)

    # Use patch.dict to temporarily replace the module in sys.modules.
    # This prevents the actual module from being imported, avoiding
    # errors if its dependencies are not installed.
    with patch.dict(sys.modules, {module_to_mock: mock_module}):
        config = Config(db_type=db_type)
        connector = get_database_connector(config)

        # Verify that the create method was called on our mock class
        mock_connector_class.create.assert_called_once_with(config)

        # Verify that the returned connector is the one from our mock
        assert connector == mock_connector_class.create.return_value


def test_get_database_connector_invalid_type():
    with pytest.raises(ValueError):
        get_database_connector(Config(db_type="InvalidDB"))
