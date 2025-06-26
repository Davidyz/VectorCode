from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.subcommands.drop import drop


@pytest.fixture
def mock_config():
    config = Config(
        project_root="/path/to/project",
    )  # Removed positional args
    return config


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def mock_collection():
    collection = AsyncMock()
    collection.name = "test_collection"
    collection.metadata = {"path": "/path/to/project"}
    return collection


@pytest.mark.asyncio
async def test_drop_success(mock_config, mock_client, mock_collection):
    mock_client.get_collection.return_value = mock_collection
    mock_client.delete_collection = AsyncMock()
    with (
        patch("vectorcode.subcommands.drop.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.drop.get_collection", return_value=mock_collection
        ),
    ):
        mock_client = AsyncMock()

        @asynccontextmanager
        async def _get_client(self, config=None, need_lock=True):
            yield mock_client

        mock_client_manager = MockClientManager.return_value
        mock_client_manager._create_client = AsyncMock(return_value=mock_client)
        mock_client_manager.get_client = _get_client

        result = await drop(mock_config)
        assert result == 0
        mock_client.delete_collection.assert_called_once_with(mock_collection.name)


@pytest.mark.asyncio
async def test_drop_collection_not_found(mock_config, mock_client):
    mock_client.get_collection.side_effect = ValueError("Collection not found")
    with patch("vectorcode.subcommands.drop.ClientManager"):
        with patch(
            "vectorcode.subcommands.drop.get_collection",
            side_effect=ValueError("Collection not found"),
        ):
            result = await drop(mock_config)
            assert result == 1
