from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import McpError

from vectorcode.cli_utils import Config
from vectorcode.mcp_main import (
    list_collections,
    ls_files,
    mcp_server,
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
    ):
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        mock_db._configs = mock_config

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
    with (
        patch("vectorcode.mcp_main.get_database_connector", return_value=mock_db),
        patch("vectorcode.mcp_main.get_project_config", return_value=mock_config),
        patch("os.path.isfile", side_effect=lambda x: x == "file1.py"),
    ):
        await vectorise_files(paths=["file1.py"], project_root=str(tmp_path))
        mock_db.vectorise.assert_called_with(file_path="file1.py")


@pytest.mark.asyncio
async def test_mcp_server():
    with (
        patch("mcp.server.fastmcp.FastMCP.add_tool") as mock_add_tool,
    ):
        await mcp_server()
        assert mock_add_tool.call_count > 0


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
    with (
        patch("vectorcode.mcp_main.get_database_connector") as mock_get_db,
        patch("vectorcode.mcp_main.get_project_config") as mock_get_config,
        patch("os.path.isfile", side_effect=lambda x: x == "file1.py"),
    ):
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        mock_get_config.return_value = Config(project_root=str(tmp_path))

        await rm_files(files=["file1.py"], project_root=str(tmp_path))

        mock_db.delete.assert_called_once()
