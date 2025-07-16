import json
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tabulate

from vectorcode.cli_utils import Config
from vectorcode.subcommands.ls import get_collection_list, ls


@pytest.mark.asyncio
async def test_get_collection_list():
    mock_client = AsyncMock()
    mock_collection1 = AsyncMock()
    mock_collection1.name = "test_collection_1"
    mock_collection1.metadata = {
        "path": "/test/path1",
        "username": "test_user",
        "embedding_function": "test_ef",
    }
    mock_collection1.count.return_value = 100
    mock_collection1.get.return_value = {
        "metadatas": [
            {"path": "/test/path1/file1.txt"},
            {"path": "/test/path1/file2.txt"},
            None,
        ]
    }
    mock_collection2 = AsyncMock()
    mock_collection2.name = "test_collection_2"
    mock_collection2.metadata = {
        "path": "/test/path2",
        "username": "test_user",
        "embedding_function": "test_ef",
    }
    mock_collection2.count.return_value = 200
    mock_collection2.get.return_value = {
        "metadatas": [
            {"path": "/test/path2/file1.txt"},
            {"path": "/test/path2/file2.txt"},
        ]
    }

    async def mock_get_collections(client):
        yield mock_collection1
        yield mock_collection2

    with patch("vectorcode.subcommands.ls.get_collections", new=mock_get_collections):
        result = await get_collection_list(mock_client)

    assert len(result) == 2
    assert result[0]["project-root"] == "/test/path1"
    assert result[0]["user"] == "test_user"
    assert result[0]["hostname"] == socket.gethostname()
    assert result[0]["collection_name"] == "test_collection_1"
    assert result[0]["size"] == 100
    assert result[0]["embedding_function"] == "test_ef"
    assert result[0]["num_files"] == 2
    assert result[1]["num_files"] == 2


@pytest.mark.asyncio
async def test_ls_pipe_mode(capsys):
    mock_client = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.name = "test_collection"
    mock_collection.metadata = {
        "path": "/test/path",
        "username": "test_user",
        "embedding_function": "test_ef",
    }
    mock_collection.count.return_value = 50
    mock_collection.get.return_value = {"metadatas": [{"path": "/test/path/file.txt"}]}

    async def mock_get_collections(client):
        yield mock_collection

    with (
        patch("vectorcode.subcommands.ls.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.ls.get_collection_list",
            return_value=[
                {
                    "project-root": "/test/path",
                    "size": 50,
                    "num_files": 1,
                    "embedding_function": "test_ef",
                }
            ],
        ),
    ):
        mock_client = MagicMock()
        mock_client_manager = MockClientManager.return_value
        mock_client_manager._create_client = AsyncMock(return_value=mock_client)

        config = Config(pipe=True)
        await ls(config)
        captured = capsys.readouterr()
        expected_output = (
            json.dumps(
                [
                    {
                        "project-root": "/test/path",
                        "size": 50,
                        "num_files": 1,
                        "embedding_function": "test_ef",
                    }
                ]
            )
            + "\n"
        )
        assert captured.out == expected_output


@pytest.mark.asyncio
async def test_ls_table_mode(capsys, monkeypatch):
    mock_client = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.name = "test_collection"
    mock_collection.metadata = {
        "path": "/test/path",
        "username": "test_user",
        "embedding_function": "test_ef",
    }
    mock_collection.count.return_value = 50
    mock_collection.get.return_value = {"metadatas": [{"path": "/test/path/file.txt"}]}

    async def mock_get_collections(client):
        yield mock_collection

    with (
        patch("vectorcode.subcommands.ls.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.ls.get_collection_list",
            return_value=[
                {
                    "project-root": "/test/path",
                    "size": 50,
                    "num_files": 1,
                    "embedding_function": "test_ef",
                }
            ],
        ),
    ):
        mock_client = MagicMock()
        mock_client_manager = MockClientManager.return_value
        mock_client_manager._create_client = AsyncMock(return_value=mock_client)

        config = Config(pipe=False)
        await ls(config)
        captured = capsys.readouterr()
        expected_output = (
            tabulate.tabulate(
                [["/test/path", 50, 1, "test_ef"]],
                headers=[
                    "Project Root",
                    "Collection Size",
                    "Number of Files",
                    "Embedding Function",
                ],
            )
            + "\n"
        )
        assert captured.out == expected_output

    # Test with HOME environment variable set
    monkeypatch.setenv("HOME", "/test")
    with (
        patch("vectorcode.subcommands.ls.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.ls.get_collection_list",
            return_value=[
                {
                    "project-root": "/test/path",
                    "size": 50,
                    "num_files": 1,
                    "embedding_function": "test_ef",
                }
            ],
        ),
    ):
        mock_client = MagicMock()
        mock_client_manager = MockClientManager.return_value
        mock_client_manager._create_client = AsyncMock(return_value=mock_client)
        config = Config(pipe=False)
        await ls(config)
        captured = capsys.readouterr()
        expected_output = (
            tabulate.tabulate(
                [["~/path", 50, 1, "test_ef"]],
                headers=[
                    "Project Root",
                    "Collection Size",
                    "Number of Files",
                    "Embedding Function",
                ],
            )
            + "\n"
        )
        assert captured.out == expected_output
