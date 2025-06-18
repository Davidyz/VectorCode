from argparse import ArgumentParser
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import McpError

from vectorcode.cli_utils import Config
from vectorcode.mcp_main import (
    get_arg_parser,
    list_collections,
    mcp_server,
    parse_cli_args,
    query_tool,
)


@pytest.mark.asyncio
async def test_list_collections_success():
    with (
        patch("vectorcode.mcp_main.get_client") as mock_get_client,
        patch("vectorcode.mcp_main.get_collections") as mock_get_collections,
    ):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        mock_collection1 = AsyncMock()
        mock_collection1.metadata = {"path": "path1"}
        mock_collection2 = AsyncMock()
        mock_collection2.metadata = {"path": "path2"}

        async def async_generator():
            yield mock_collection1
            yield mock_collection2

        mock_get_collections.return_value = async_generator()

        result = await list_collections()
        assert result == ["path1", "path2"]


@pytest.mark.asyncio
async def test_list_collections_no_metadata():
    with (
        patch("vectorcode.mcp_main.get_client") as mock_get_client,
        patch("vectorcode.mcp_main.get_collections") as mock_get_collections,
    ):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_collection1 = AsyncMock()
        mock_collection1.metadata = {"path": "path1"}
        mock_collection2 = AsyncMock()
        mock_collection2.metadata = None

        async def async_generator(cli):
            yield mock_collection1
            yield mock_collection2

        mock_get_collections.side_effect = async_generator

        result = await list_collections()
        assert result == ["path1"]


@pytest.mark.asyncio
async def test_query_tool_invalid_project_root():
    with pytest.raises(McpError) as exc_info:
        await query_tool(
            n_query=5,
            query_messages=["keyword1", "keyword2"],
            project_root="invalid_path",
        )
    assert exc_info.value.error.code == 1
    assert (
        exc_info.value.error.message
        == "Use `list_collections` tool to get a list of valid paths for this field."
    )


@pytest.mark.asyncio
async def test_query_tool_success():
    with (
        patch("os.path.isdir", return_value=True),
        patch("vectorcode.mcp_main.get_project_config") as mock_get_project_config,
        patch("vectorcode.mcp_main.get_client") as mock_get_client,
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
        patch(
            "vectorcode.subcommands.query.get_query_result_files"
        ) as mock_get_query_result_files,
        patch("builtins.open", create=True) as mock_open,
        patch("os.path.isfile", return_value=True),
        patch("os.path.relpath", return_value="rel/path.py"),
        patch("vectorcode.cli_utils.load_config_file") as mock_load_config_file,
    ):
        mock_config = Config(chunk_size=100, overlap_ratio=0.1, reranker=None)
        mock_load_config_file.return_value = mock_config
        mock_get_project_config.return_value = mock_config
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Mock the collection's query method to return a valid QueryResult
        mock_collection = AsyncMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "embeddings": None,
            "metadatas": [[{"path": "file1.py"}, {"path": "file2.py"}]],
            "documents": [["doc1", "doc2"]],
            "uris": None,
            "data": None,
            "distances": [[0.1, 0.2]],  # Valid distances
        }
        mock_get_collection.return_value = mock_collection

        mock_get_query_result_files.return_value = ["file1.py", "file2.py"]
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = "file content"
        mock_open.return_value = mock_file_handle

        result = await query_tool(
            n_query=2, query_messages=["keyword1"], project_root="/valid/path"
        )

        assert len(result) == 2
        assert "<path>rel/path.py</path>\n<content>file content</content>" in result


@pytest.mark.asyncio
async def test_query_tool_collection_access_failure():
    with (
        patch("os.path.isdir", return_value=True),
        patch("vectorcode.mcp_main.get_project_config"),
        patch("vectorcode.mcp_main.get_client") as mock_get_client,
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
    ):
        mock_get_client.side_effect = Exception("Failed to connect")
        mock_get_collection.side_effect = Exception("Failed to connect")

        with pytest.raises(McpError) as exc_info:
            await query_tool(
                n_query=2, query_messages=["keyword1"], project_root="/valid/path"
            )

        assert exc_info.value.error.code == 1
        assert (
            "Failed to access the collection at /valid/path. Use `list_collections` tool to get a list of valid paths for this field."
            in exc_info.value.error.message
        )


@pytest.mark.asyncio
async def test_query_tool_no_collection():
    with (
        patch("os.path.isdir", return_value=True),
        patch("vectorcode.mcp_main.get_project_config"),
        patch("vectorcode.mcp_main.get_client"),
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
    ):
        mock_get_collection.return_value = None

        with pytest.raises(McpError) as exc_info:
            await query_tool(
                n_query=2, query_messages=["keyword1"], project_root="/valid/path"
            )

        assert exc_info.value.error.code == 1
        assert (
            exc_info.value.error.message
            == "Failed to access the collection at /valid/path. Use `list_collections` tool to get a list of valid paths for this field."
        )


@pytest.mark.asyncio
async def test_mcp_server():
    with (
        patch(
            "vectorcode.mcp_main.find_project_config_dir"
        ) as mock_find_project_config_dir,
        patch("vectorcode.mcp_main.load_config_file") as mock_load_config_file,
        patch("vectorcode.mcp_main.get_client") as mock_get_client,
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
        patch("mcp.server.fastmcp.FastMCP.add_tool") as mock_add_tool,
    ):
        mock_find_project_config_dir.return_value = "/path/to/config"
        mock_load_config_file.return_value = Config(project_root="/path/to/project")
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_collection = AsyncMock()
        mock_get_collection.return_value = mock_collection

        await mcp_server()

        assert mock_add_tool.call_count == 2


@pytest.mark.asyncio
async def test_mcp_server_ls_on_start():
    with (
        patch(
            "vectorcode.mcp_main.find_project_config_dir"
        ) as mock_find_project_config_dir,
        patch("vectorcode.mcp_main.load_config_file") as mock_load_config_file,
        patch("vectorcode.mcp_main.get_client") as mock_get_client,
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
        patch(
            "vectorcode.mcp_main.get_collections", spec=AsyncMock
        ) as mock_get_collections,
        patch("mcp.server.fastmcp.FastMCP.add_tool") as mock_add_tool,
    ):
        from vectorcode.mcp_main import mcp_config

        mcp_config.ls_on_start = True
        mock_find_project_config_dir.return_value = "/path/to/config"
        mock_load_config_file.return_value = Config(project_root="/path/to/project")
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_collection = AsyncMock()
        mock_collection.metadata = {"path": "/path/to/project"}
        mock_get_collection.return_value = mock_collection

        async def new_get_collections(clients):
            yield mock_collection

        mock_get_collections.side_effect = new_get_collections

        await mcp_server()

        assert mock_add_tool.call_count == 2
        mock_get_collections.assert_called()


def test_arg_parser():
    assert isinstance(get_arg_parser(), ArgumentParser)


def test_args_parsing():
    args = ["--number", "15", "--ls-on-start"]
    parsed = parse_cli_args(args)
    assert parsed.n_results == 15
    assert parsed.ls_on_start
