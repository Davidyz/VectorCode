import json
from unittest.mock import AsyncMock, patch

import pytest
import tabulate

from vectorcode.cli_utils import Config
from vectorcode.database.types import CollectionInfo
from vectorcode.subcommands.ls import ls


@pytest.fixture
def mock_collections():
    return [
        CollectionInfo(
            path="/test/path1",
            id="test_collection_1",
            chunk_count=100,
            file_count=2,
            embedding_function="test_ef",
            database_backend="ChromaDB",
        ),
        CollectionInfo(
            path="/test/path2",
            id="test_collection_2",
            chunk_count=200,
            file_count=2,
            embedding_function="test_ef",
            database_backend="ChromaDB",
        ),
    ]


@pytest.mark.asyncio
async def test_ls_pipe_mode(capsys, mock_collections):
    mock_db = AsyncMock()
    mock_db.list_collections.return_value = mock_collections
    with patch(
        "vectorcode.subcommands.ls.get_database_connector", return_value=mock_db
    ):
        config = Config(pipe=True)
        await ls(config)
        captured = capsys.readouterr()
        expected_output = json.dumps([c.to_dict() for c in mock_collections]) + "\n"
        assert captured.out == expected_output


@pytest.mark.asyncio
async def test_ls_table_mode(capsys, mock_collections, monkeypatch):
    mock_db = AsyncMock()
    mock_db.list_collections.return_value = mock_collections
    with patch(
        "vectorcode.subcommands.ls.get_database_connector", return_value=mock_db
    ):
        config = Config(pipe=False)
        await ls(config)
        captured = capsys.readouterr()
        expected_table = [
            ["/test/path1", 100, 2, "test_ef"],
            ["/test/path2", 200, 2, "test_ef"],
        ]
        expected_output = (
            tabulate.tabulate(
                expected_table,
                headers=[
                    "Project Root",
                    "Number of Embeddings",
                    "Number of Files",
                    "Embedding Function",
                ],
            )
            + "\n"
        )
        assert captured.out == expected_output

    # Test with HOME environment variable set
    monkeypatch.setenv("HOME", "/test")
    with patch(
        "vectorcode.subcommands.ls.get_database_connector", return_value=mock_db
    ):
        config = Config(pipe=False)
        await ls(config)
        captured = capsys.readouterr()
        expected_table = [
            ["~/path1", 100, 2, "test_ef"],
            ["~/path2", 200, 2, "test_ef"],
        ]
        expected_output = (
            tabulate.tabulate(
                expected_table,
                headers=[
                    "Project Root",
                    "Number of Embeddings",
                    "Number of Files",
                    "Embedding Function",
                ],
            )
            + "\n"
        )
        assert captured.out == expected_output
