from unittest.mock import AsyncMock, patch

import pytest
from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException

from vectorcode.cli_utils import Config
from vectorcode.subcommands.update import update


@pytest.mark.asyncio
async def test_update_success():
    mock_client = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {
        "metadatas": [{"path": "file1.py"}, {"path": "file2.py"}]
    }
    mock_collection.delete = AsyncMock()
    mock_client.get_max_batch_size.return_value = 100

    with (
        patch("vectorcode.subcommands.update.ClientManager"),
        patch(
            "vectorcode.subcommands.update.get_collection", return_value=mock_collection
        ),
        patch("vectorcode.subcommands.update.verify_ef", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch(
            "vectorcode.subcommands.update.chunked_add", new_callable=AsyncMock
        ) as mock_chunked_add,
        patch("vectorcode.subcommands.update.show_stats"),
    ):
        config = Config(project_root="/test/project", pipe=False)
        result = await update(config)

        assert result == 0
        mock_collection.get.assert_called_once_with(include=[IncludeEnum.metadatas])
        assert mock_chunked_add.call_count == 2
        mock_collection.delete.assert_not_called()


@pytest.mark.asyncio
async def test_update_with_orphans():
    mock_client = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {
        "metadatas": [{"path": "file1.py"}, {"path": "file2.py"}, {"path": "orphan.py"}]
    }
    mock_collection.delete = AsyncMock()
    mock_client.get_max_batch_size.return_value = 100

    with (
        patch("vectorcode.subcommands.update.ClientManager"),
        patch(
            "vectorcode.subcommands.update.get_collection", return_value=mock_collection
        ),
        patch("vectorcode.subcommands.update.verify_ef", return_value=True),
        patch("os.path.isfile", side_effect=[True, True, False]),
        patch(
            "vectorcode.subcommands.update.chunked_add", new_callable=AsyncMock
        ) as mock_chunked_add,
        patch("vectorcode.subcommands.update.show_stats"),
    ):
        config = Config(project_root="/test/project", pipe=False)
        result = await update(config)

        assert result == 0
        mock_collection.get.assert_called_once_with(include=[IncludeEnum.metadatas])
        assert mock_chunked_add.call_count == 2
        mock_collection.delete.assert_called_once_with(
            where={"path": {"$in": ["orphan.py"]}}
        )


@pytest.mark.asyncio
async def test_update_index_error():
    mock_client = AsyncMock()
    # mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.update.ClientManager") as MockClientManager,
        patch("vectorcode.subcommands.update.get_collection", side_effect=IndexError),
        patch("sys.stderr"),
    ):
        MockClientManager.return_value._create_client.return_value = mock_client
        config = Config(project_root="/test/project", pipe=False)
        result = await update(config)

        assert result == 1


@pytest.mark.asyncio
async def test_update_value_error():
    mock_client = AsyncMock()
    # mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.update.ClientManager") as MockClientManager,
        patch("vectorcode.subcommands.update.get_collection", side_effect=ValueError),
        patch("sys.stderr"),
    ):
        MockClientManager.return_value._create_client.return_value = mock_client
        config = Config(project_root="/test/project", pipe=False)
        result = await update(config)

        assert result == 1


@pytest.mark.asyncio
async def test_update_invalid_collection_exception():
    mock_client = AsyncMock()
    # mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.update.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.update.get_collection",
            side_effect=InvalidCollectionException,
        ),
        patch("sys.stderr"),
    ):
        MockClientManager.return_value._create_client.return_value = mock_client
        config = Config(project_root="/test/project", pipe=False)
        result = await update(config)

        assert result == 1
