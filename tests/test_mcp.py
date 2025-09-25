import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from mcp import ErrorData, McpError

from vectorcode.cli_utils import Config
from vectorcode.mcp_main import (
    get_arg_parser,
    list_collections,
    ls_files,
    mcp_config,
    mcp_server,
    parse_cli_args,
    query_tool,
    rm_files,
    vectorise_files,
)


@pytest.mark.asyncio
async def test_list_collections_success():
    with patch("vectorcode.mcp_main.get_database_connector") as mock_get_db:
        mock_db = AsyncMock()
        mock_db.list_collections.return_value = [
            MagicMock(path="path1"),
            MagicMock(path="path2"),
        ]
        mock_get_db.return_value = mock_db

        result = await list_collections()
        assert result == ["path1", "path2"]


@pytest.mark.asyncio
async def test_query_tool_invalid_project_root():
    with patch("os.path.isdir", return_value=False):
        with pytest.raises(McpError) as exc_info:
            await query_tool(
                n_query=5,
                query_messages=["keyword1", "keyword2"],
                project_root="invalid_path",
            )
        assert exc_info.value.error.code == 1


@pytest.mark.asyncio
async def test_query_tool_success(tmp_path):
    mock_config = Config(project_root=tmp_path)
    with (
        patch("vectorcode.mcp_main.get_database_connector") as mock_get_db,
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch(
            "vectorcode.subcommands.query.reranker.naive.NaiveReranker.rerank",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        mock_db._configs = mock_config
        mock_db.query.return_value = []

        await query_tool(
            n_query=2, query_messages=["keyword1"], project_root=str(tmp_path)
        )
        mock_db.query.assert_called_once()
        assert mock_db._configs.n_result == 2
        assert mock_db._configs.query == ["keyword1"]


@pytest.mark.asyncio
async def test_vectorise_tool_invalid_project_root():
    with patch("os.path.isdir", return_value=False):
        with pytest.raises(McpError):
            await vectorise_files(paths=["foo.bar"], project_root=".")


@pytest.mark.asyncio
async def test_vectorise_files_success(tmp_path):
    mock_db = AsyncMock()
    mock_config = Config(project_root=str(tmp_path))
    (tmp_path / "file1.py").touch()
    with (
        patch("vectorcode.mcp_main.get_database_connector", return_value=mock_db),
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch(
            "vectorcode.mcp_main.vectorise_worker", new_callable=AsyncMock
        ) as mock_worker,
    ):
        await vectorise_files(
            paths=[str(tmp_path / "file1.py")], project_root=str(tmp_path)
        )
        mock_worker.assert_called_once()


@pytest.mark.asyncio
async def test_vectorise_files_with_ignore_spec(tmp_path):
    project_root = tmp_path
    (project_root / ".gitignore").write_text("ignored.py")
    (project_root / "file1.py").touch()
    (project_root / "ignored.py").touch()

    mock_db = AsyncMock()
    mock_config = Config(project_root=str(project_root))
    with (
        patch("vectorcode.mcp_main.get_database_connector", return_value=mock_db),
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch(
            "vectorcode.mcp_main.vectorise_worker", new_callable=AsyncMock
        ) as mock_worker,
    ):
        await vectorise_files(
            paths=[str(project_root / "file1.py"), str(project_root / "ignored.py")],
            project_root=str(project_root),
        )
        mock_worker.assert_called_once_with(
            mock_db, str(project_root / "file1.py"), ANY, ANY, ANY
        )


@pytest.mark.asyncio
async def test_mcp_server(tmp_path):
    with (
        patch("mcp.server.fastmcp.FastMCP.add_tool") as mock_add_tool,
        patch("vectorcode.mcp_main.find_project_config_dir", return_value=tmp_path),
        patch("vectorcode.mcp_main.get_project_config", return_value=Config()),
    ):
        await mcp_server()
        assert mock_add_tool.call_count > 0


@pytest.mark.asyncio
async def test_mcp_server_ls_on_start(tmp_path):
    with (
        patch("mcp.server.fastmcp.FastMCP.add_tool") as mock_add_tool,
        patch("vectorcode.mcp_main.find_project_config_dir", return_value=tmp_path),
        patch("vectorcode.mcp_main.get_project_config", return_value=Config()),
        patch("vectorcode.mcp_main.list_collections", return_value=["path1", "path2"]),
    ):
        mcp_config.ls_on_start = True
        await mcp_server()
        assert mock_add_tool.call_count > 0
        mcp_config.ls_on_start = False


@pytest.mark.asyncio
async def test_ls_files_success(tmp_path):
    with (
        patch("vectorcode.mcp_main.get_database_connector") as mock_get_db,
        patch("vectorcode.mcp_main.get_project_config") as mock_get_config,
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.return_value.files = [
            MagicMock(path="file1.py"),
            MagicMock(path="file2.py"),
        ]
        mock_get_db.return_value = mock_db
        mock_get_config.return_value = Config(project_root=str(tmp_path))

        result = await ls_files(project_root=str(tmp_path))

        assert result == ["file1.py", "file2.py"]


@pytest.mark.asyncio
async def test_rm_files_success(tmp_path):
    (tmp_path / "file1.py").touch()
    with (
        patch("vectorcode.mcp_main.get_database_connector") as mock_get_db,
        patch("vectorcode.mcp_main.get_project_config") as mock_get_config,
    ):
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        mock_get_config.return_value = Config(project_root=str(tmp_path))

        await rm_files(files=[str(tmp_path / "file1.py")], project_root=str(tmp_path))

        mock_db.delete.assert_called_once()


@pytest.mark.asyncio
async def test_rm_files_no_files(tmp_path):
    with (
        patch("vectorcode.mcp_main.get_database_connector") as mock_get_db,
        patch("vectorcode.mcp_main.get_project_config") as mock_get_config,
    ):
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        mock_get_config.return_value = Config(project_root=str(tmp_path))

        await rm_files(files=["file1.py"], project_root=str(tmp_path))

        mock_db.delete.assert_not_called()


def test_get_arg_parser():
    parser = get_arg_parser()
    args = parser.parse_args(["-n", "5", "--ls-on-start"])
    assert args.number == 5
    assert args.ls_on_start is True


def test_parse_cli_args():
    with patch.object(sys, "argv", ["", "-n", "5", "--ls-on-start"]):
        config = parse_cli_args()
        assert config.n_results == 5
        assert config.ls_on_start is True


@pytest.mark.asyncio
async def test_vectorise_files_exception(tmp_path):
    mock_db = AsyncMock()
    mock_config = Config(project_root=str(tmp_path))
    (tmp_path / "file1.py").touch()
    with (
        patch("vectorcode.mcp_main.get_database_connector", return_value=mock_db),
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch(
            "vectorcode.mcp_main.vectorise_worker", side_effect=Exception("test error")
        ),
    ):
        with pytest.raises(McpError):
            await vectorise_files(
                paths=[str(tmp_path / "file1.py")], project_root=str(tmp_path)
            )


@pytest.mark.asyncio
async def test_query_tool_exception(tmp_path):
    mock_config = Config(project_root=tmp_path)
    with (
        patch("vectorcode.mcp_main.get_database_connector") as mock_get_db,
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch(
            "vectorcode.mcp_main.get_reranked_results",
            side_effect=Exception("test error"),
        ),
    ):
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        mock_db._configs = mock_config

        with pytest.raises(McpError):
            await query_tool(
                n_query=2, query_messages=["keyword1"], project_root=str(tmp_path)
            )


@pytest.mark.asyncio
async def test_vectorise_files_mcp_exception(tmp_path):
    mock_db = AsyncMock()
    mock_config = Config(project_root=str(tmp_path))
    (tmp_path / "file1.py").touch()
    with (
        patch("vectorcode.mcp_main.get_database_connector", return_value=mock_db),
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch(
            "vectorcode.mcp_main.vectorise_worker",
            side_effect=McpError(ErrorData(code=1, message="test error")),
        ),
    ):
        with pytest.raises(McpError):
            await vectorise_files(
                paths=[str(tmp_path / "file1.py")], project_root=str(tmp_path)
            )


@pytest.mark.asyncio
async def test_query_tool_mcp_exception(tmp_path):
    mock_config = Config(project_root=tmp_path)
    with (
        patch("vectorcode.mcp_main.get_database_connector") as mock_get_db,
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch(
            "vectorcode.mcp_main.get_reranked_results",
            side_effect=McpError(ErrorData(code=1, message="test error")),
        ),
    ):
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        mock_db._configs = mock_config

        with pytest.raises(McpError):
            await query_tool(
                n_query=2, query_messages=["keyword1"], project_root=str(tmp_path)
            )
