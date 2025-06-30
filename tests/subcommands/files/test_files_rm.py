from unittest.mock import AsyncMock, patch

import pytest
from chromadb.api.models.AsyncCollection import AsyncCollection

from vectorcode.cli_utils import CliAction, Config, FilesAction
from vectorcode.subcommands.files.rm import rm


@pytest.fixture
def client():
    return AsyncMock()


@pytest.fixture
def collection():
    col = AsyncMock(spec=AsyncCollection)
    col.get.return_value = {
        "ids": ["id1", "id2", "id3"],
        "distances": [0.1, 0.2, 0.3],
        "metadatas": [
            {"path": "file1.py", "start": 1, "end": 1},
            {"path": "file2.py", "start": 1, "end": 1},
            {"path": "file3.py", "start": 1, "end": 1},
        ],
        "documents": [
            "content1",
            "content2",
            "content3",
        ],
    }
    col.name = "test_collection"
    return col


@pytest.mark.asyncio
async def test_rm(client, collection, capsys):
    with (
        patch("vectorcode.subcommands.files.rm.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.files.rm.get_collection", return_value=collection
        ),
        patch("vectorcode.common.try_server", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch(
            "vectorcode.subcommands.files.rm.expand_path", side_effect=lambda x, y: x
        ),
    ):
        MockClientManager.return_value._create_client.return_value = client
        config = Config(
            action=CliAction.files,
            files_action=FilesAction.rm,
            rm_paths=["file1.py"],
        )
        await rm(config)
        collection.delete.assert_called_with(where={"path": {"$in": ["file1.py"]}})


@pytest.mark.asyncio
async def test_rm_empty_collection(client, collection, capsys):
    with (
        patch(
            "vectorcode.subcommands.files.rm.get_collection", return_value=collection
        ),
        patch("vectorcode.common.try_server", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch(
            "vectorcode.subcommands.files.rm.expand_path", side_effect=lambda x, y: x
        ),
        patch(
            "vectorcode.subcommands.files.rm.ClientManager._create_client",
            return_value=client,
        ),
    ):
        config = Config(
            action=CliAction.files,
            files_action=FilesAction.rm,
            rm_paths=["file1.py"],
        )
        collection.count = AsyncMock(return_value=0)
        client.delete_collection = AsyncMock()
        await rm(config)
        client.delete_collection.assert_called_once_with(collection.name)


@pytest.mark.asyncio
async def test_rm_no_collection(client, collection, capsys):
    with (
        patch("vectorcode.subcommands.files.rm.ClientManager") as MockClientManager,
        patch("vectorcode.subcommands.files.rm.get_collection", side_effect=ValueError),
    ):
        MockClientManager.return_value._create_client.return_value = client
        assert (
            await rm(
                Config(
                    action=CliAction.files,
                    files_action=FilesAction.rm,
                    pipe=True,
                    rm_paths=["file1.py"],
                )
            )
            != 0
        )
