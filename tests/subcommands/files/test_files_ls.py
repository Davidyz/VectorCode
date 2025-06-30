import json
from unittest.mock import AsyncMock, patch

import pytest
from chromadb.api.models.AsyncCollection import AsyncCollection

from vectorcode.cli_utils import CliAction, Config, FilesAction
from vectorcode.subcommands.files.ls import ls


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
    return col


@pytest.mark.asyncio
async def test_ls(client, collection, capsys):
    with (
        patch("vectorcode.subcommands.files.ls.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.files.ls.get_collection", return_value=collection
        ),
        patch("vectorcode.common.try_server", return_value=True),
    ):
        MockClientManager.return_value._create_client.return_value = client
        await ls(Config(action=CliAction.files, files_action=FilesAction.ls))
        out = capsys.readouterr().out
        assert "file1.py" in out
        assert "file2.py" in out
        assert "file3.py" in out


@pytest.mark.asyncio
async def test_ls_piped(client, collection, capsys):
    with (
        patch("vectorcode.subcommands.files.ls.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.files.ls.get_collection", return_value=collection
        ),
        patch("vectorcode.common.try_server", return_value=True),
    ):
        MockClientManager.return_value._create_client.return_value = client
        await ls(Config(action=CliAction.files, files_action=FilesAction.ls, pipe=True))
        out = capsys.readouterr().out
        assert json.dumps(["file1.py", "file2.py", "file3.py"]).strip() == out.strip()


@pytest.mark.asyncio
async def test_ls_no_collection(client, collection, capsys):
    with (
        patch("vectorcode.subcommands.files.ls.ClientManager") as MockClientManager,
        patch("vectorcode.subcommands.files.ls.get_collection", side_effect=ValueError),
    ):
        MockClientManager.return_value._create_client.return_value = client
        assert (
            await ls(
                Config(action=CliAction.files, files_action=FilesAction.ls, pipe=True)
            )
            != 0
        )


@pytest.mark.asyncio
async def test_ls_empty_collection(client, capsys):
    mock_collection = AsyncMock(spec=AsyncCollection)
    mock_collection.get.return_value = {}
    with (
        patch("vectorcode.subcommands.files.ls.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.files.ls.get_collection",
            return_value=mock_collection,
        ),
        patch("vectorcode.common.try_server", return_value=True),
    ):
        MockClientManager.return_value._create_client.return_value = client
        assert (
            await ls(Config(action=CliAction.files, files_action=FilesAction.ls)) == 0
        )
