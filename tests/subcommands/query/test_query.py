from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tree_sitter import Point

from vectorcode.chunking import Chunk
from vectorcode.cli_utils import CliAction, Config, QueryInclude
from vectorcode.database.base import DatabaseConnectorBase
from vectorcode.subcommands.query import query


@pytest.fixture
def mock_config():
    return Config(
        action=CliAction.query,
        query=["test query"],
        n_result=3,
        project_root="/test/project",
        pipe=False,
        include=[QueryInclude.path, QueryInclude.document],
        query_exclude=[],
        reranker="NaiveReranker",
        reranker_params={},
        use_absolute_path=False,
    )


@pytest.fixture
def mock_database():
    db = AsyncMock(spec=DatabaseConnectorBase)
    db.query.return_value = [
        MagicMock(path="file1.py", document="content1"),
        MagicMock(path="file2.py", document="content2"),
    ]
    return db


@pytest.mark.asyncio
async def test_query_success(mock_config, mock_database, capsys):
    with (
        patch(
            "vectorcode.subcommands.query.get_database_connector",
            return_value=mock_database,
        ),
        patch("vectorcode.subcommands.query.get_reranker") as mock_get_reranker,
    ):
        mock_reranker = AsyncMock()
        mock_reranker.rerank.return_value = [
            "file1.py",
            "file2.py",
        ]
        mock_get_reranker.return_value = mock_reranker

        with (
            patch("builtins.open", MagicMock()),
            patch("os.path.isfile", return_value=True),
        ):
            result = await query(mock_config)

        assert result == 0
        captured = capsys.readouterr()
        assert "Path: file1.py" in captured.out
        assert "Path: file2.py" in captured.out


@pytest.mark.asyncio
async def test_query_pipe_mode(mock_config, mock_database):
    mock_config.pipe = True
    with (
        patch(
            "vectorcode.subcommands.query.get_database_connector",
            return_value=mock_database,
        ),
        patch("vectorcode.subcommands.query.get_reranker") as mock_get_reranker,
        patch("json.dumps") as mock_json_dumps,
    ):
        mock_reranker = AsyncMock()
        mock_reranker.rerank.return_value = [
            "file1.py",
            "file2.py",
        ]
        mock_get_reranker.return_value = mock_reranker

        with (
            patch("builtins.open", MagicMock()),
            patch("os.path.isfile", return_value=True),
        ):
            await query(mock_config)

        mock_json_dumps.assert_called_once()


@pytest.mark.asyncio
async def test_query_chunk_mode(mock_config, mock_database, capsys):
    mock_config.include = [QueryInclude.chunk]
    chunk1 = Chunk(text="chunk1", path="file1.py", start=Point(1, 0), end=Point(2, 0))
    chunk2 = Chunk(text="chunk2", path="file1.py", start=Point(3, 0), end=Point(4, 0))

    with (
        patch(
            "vectorcode.subcommands.query.get_database_connector",
            return_value=mock_database,
        ),
        patch("vectorcode.subcommands.query.get_reranker") as mock_get_reranker,
    ):
        mock_reranker = AsyncMock()
        mock_reranker.rerank.return_value = [chunk1, chunk2]
        mock_get_reranker.return_value = mock_reranker

        await query(mock_config)
        captured = capsys.readouterr()
        assert "Chunk: chunk1" in captured.out
        assert "Chunk: chunk2" in captured.out


@pytest.mark.asyncio
async def test_query_invalid_include(mock_config):
    mock_config.include = [QueryInclude.chunk, QueryInclude.document]
    result = await query(mock_config)
    assert result != 0
