from unittest.mock import AsyncMock, patch

import pytest
from chromadb.api import AsyncClientAPI

from vectorcode.cli_utils import Config
from vectorcode.subcommands.clean import clean, run_clean_on_client


@pytest.mark.asyncio
async def test_run_clean_on_client():
    mock_client = AsyncMock(spec=AsyncClientAPI)
    mock_collection1 = AsyncMock()
    mock_collection1.name = "test_collection_1"
    mock_collection1.metadata = {"path": "/test/path1"}
    mock_collection1.count.return_value = 0  # Empty collection
    mock_collection2 = AsyncMock()
    mock_collection2.name = "test_collection_2"
    mock_collection2.metadata = {"path": "/test/path2"}
    mock_collection2.count.return_value = 1  # Non-empty collection

    async def mock_get_collections(client):
        yield mock_collection1
        yield mock_collection2

    with (
        patch("vectorcode.subcommands.clean.get_collections", new=mock_get_collections),
        patch("os.path.isdir", return_value=lambda x: x == "/test/path2"),
    ):
        await run_clean_on_client(mock_client, pipe_mode=False)

    mock_client.delete_collection.assert_called_once_with(mock_collection1.name)


@pytest.mark.asyncio
async def test_run_clean_on_client_pipe_mode():
    mock_client = AsyncMock(spec=AsyncClientAPI)
    mock_collection1 = AsyncMock()
    mock_collection1.name = "test_collection_1"
    mock_collection1.metadata = {"path": "/test/path1"}
    mock_collection1.count.return_value = 0  # Empty collection

    async def mock_get_collections(client):
        yield mock_collection1

    with patch(
        "vectorcode.subcommands.clean.get_collections", new=mock_get_collections
    ):
        await run_clean_on_client(mock_client, pipe_mode=True)

    mock_client.delete_collection.assert_called_once_with(mock_collection1.name)


@pytest.mark.asyncio
async def test_run_clean_on_removed_dir():
    mock_client = AsyncMock(spec=AsyncClientAPI)
    mock_collection1 = AsyncMock()
    mock_collection1.name = "test_collection_1"
    mock_collection1.metadata = {"path": "/test/path1"}
    mock_collection1.count.return_value = 10

    async def mock_get_collections(client):
        yield mock_collection1

    with (
        patch("vectorcode.subcommands.clean.get_collections", new=mock_get_collections),
        patch("os.path.isdir", return_value=False),
    ):
        await run_clean_on_client(mock_client, pipe_mode=True)

    mock_client.delete_collection.assert_called_once_with(mock_collection1.name)


@pytest.mark.asyncio
async def test_clean():
    AsyncMock(spec=AsyncClientAPI)
    mock_config = Config(pipe=False)

    with patch("vectorcode.subcommands.clean.ClientManager"):
        result = await clean(mock_config)

    assert result == 0
