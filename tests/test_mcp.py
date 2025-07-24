import os
import tempfile
from argparse import ArgumentParser
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import numpy
import pytest
from mcp import McpError

from vectorcode.cli_utils import Config
from vectorcode.common import ClientManager
from vectorcode.mcp_main import (
    get_arg_parser,
    list_collections,
    ls_files,
    mcp_server,
    parse_cli_args,
    query_tool,
    rm_files,
    vectorise_files,
)


@pytest.mark.asyncio
async def test_list_collections_success():
    mock_client = AsyncMock()
    with (
        patch("vectorcode.mcp_main.get_collections") as mock_get_collections,
        patch("vectorcode.common.try_server", return_value=True),
        patch(
            "vectorcode.mcp_main.ClientManager._create_client", return_value=mock_client
        ),
    ):
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
    mock_client = AsyncMock()
    with (
        patch("vectorcode.mcp_main.get_collections") as mock_get_collections,
        patch("vectorcode.common.try_server", return_value=True),
        patch(
            "vectorcode.mcp_main.ClientManager._create_client", return_value=mock_client
        ),
    ):
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
    mock_client = AsyncMock()

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
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
        for i in range(1, 3):
            with open(os.path.join(temp_dir, f"file{i}.py"), "w") as fin:
                fin.writelines([f"doc{i}"])
        with (
            patch("vectorcode.mcp_main.get_project_config") as mock_get_project_config,
            patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
            patch(
                "vectorcode.mcp_main.ClientManager._create_client",
                return_value=mock_client,
            ),
            patch(
                "vectorcode.subcommands.query.get_query_result_files"
            ) as mock_get_query_result_files,
            patch("vectorcode.common.try_server", return_value=True),
            patch("vectorcode.cli_utils.load_config_file") as mock_load_config_file,
        ):
            mock_config = Config(
                chunk_size=100, overlap_ratio=0.1, reranker=None, project_root=temp_dir
            )
            mock_load_config_file.return_value = mock_config
            mock_get_project_config.return_value = mock_config

            mock_get_collection.return_value = mock_collection

            mock_get_query_result_files.return_value = [
                os.path.join(temp_dir, i) for i in ("file1.py", "file2.py")
            ]

            result = await query_tool(
                n_query=2, query_messages=["keyword1"], project_root=temp_dir
            )

            assert len(result) == 2


@pytest.mark.asyncio
async def test_query_tool_collection_access_failure():
    with (
        patch("os.path.isdir", return_value=True),
        patch("vectorcode.mcp_main.get_project_config"),
        patch("vectorcode.mcp_main.get_collection"),
        patch(
            "vectorcode.mcp_main.ClientManager._create_client",
            side_effect=Exception("Failed to connect"),
        ),
    ):
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
    mock_client = AsyncMock()
    with (
        patch("os.path.isdir", return_value=True),
        patch("vectorcode.mcp_main.get_project_config"),
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
        patch(
            "vectorcode.mcp_main.ClientManager._create_client", return_value=mock_client
        ),
    ):
        mock_get_collection.return_value = None

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
async def test_vectorise_tool_invalid_project_root():
    with (
        patch("os.path.isdir", return_value=False),
    ):
        with pytest.raises(McpError):
            await vectorise_files(paths=["foo.bar"], project_root=".")


@pytest.mark.asyncio
async def test_vectorise_files_success():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/test_file.py"
        with open(file_path, "w") as f:
            f.write("def func(): pass")
        mock_client = AsyncMock()

        mock_embedding_function = MagicMock(return_value=numpy.random.random((100,)))
        with (
            patch("os.path.isdir", return_value=True),
            patch("vectorcode.mcp_main.get_project_config") as mock_get_project_config,
            patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
            patch(
                "vectorcode.mcp_main.ClientManager._create_client",
                return_value=mock_client,
            ),
            patch(
                "vectorcode.subcommands.vectorise.get_embedding_function",
                return_value=mock_embedding_function,
            ),
            patch("vectorcode.subcommands.vectorise.chunked_add"),
            patch(
                "vectorcode.subcommands.vectorise.hash_file", return_value="test_hash"
            ),
            patch("vectorcode.common.try_server", return_value=True),
        ):
            mock_config = Config(project_root=temp_dir)
            mock_get_project_config.return_value = mock_config

            mock_collection = AsyncMock()
            mock_collection.get.return_value = {"ids": [], "metadatas": []}
            mock_get_collection.return_value = mock_collection
            mock_client.get_max_batch_size.return_value = 100

            result = await vectorise_files(paths=[file_path], project_root=temp_dir)

            assert result["add"] == 1
            mock_get_project_config.assert_called_once_with(temp_dir)
            # Assert that the mocked get_collection was called with our mock_client.
            mock_get_collection.assert_called_once()


@pytest.mark.asyncio
async def test_vectorise_files_collection_access_failure():
    with (
        patch("os.path.isdir", return_value=True),
        patch("vectorcode.mcp_main.get_project_config"),
        patch(
            "vectorcode.mcp_main.ClientManager._create_client",
            side_effect=Exception("Client error"),
        ),
        patch("vectorcode.mcp_main.get_collection"),
    ):
        with pytest.raises(McpError) as exc_info:
            await vectorise_files(paths=["file.py"], project_root="/valid/path")

        assert exc_info.value.error.code == 1
        assert (
            "Failed to create the collection at /valid/path"
            in exc_info.value.error.message
        )


@pytest.mark.asyncio
async def test_vectorise_files_with_exclude_spec():
    with tempfile.TemporaryDirectory() as temp_dir:
        file1 = f"{temp_dir}/file1.py"
        excluded_file = f"{temp_dir}/excluded.py"
        exclude_spec_file = f"{temp_dir}/.vectorcode/vectorcode.exclude"

        os.makedirs(f"{temp_dir}/.vectorcode")
        with open(file1, "w") as f:
            f.write("content1")
        with open(excluded_file, "w") as f:
            f.write("content_excluded")
        with open(exclude_spec_file, "w") as fin:
            fin.writelines(["excluded.py"])

        # Create mock file handles for specific file contents
        mock_exclude_file_handle = mock_open(read_data="excluded.py").return_value

        def mock_open_side_effect(filename, *args, **kwargs):
            if filename == exclude_spec_file:
                return mock_exclude_file_handle
            # For other files that might be opened, return a generic mock
            return MagicMock()

        mock_client = AsyncMock()
        with (
            patch("vectorcode.mcp_main.get_project_config") as mock_get_project_config,
            patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
            patch(
                "vectorcode.mcp_main.ClientManager._create_client",
                return_value=mock_client,
            ),
            patch("vectorcode.mcp_main.chunked_add") as mock_chunked_add,
            patch(
                "vectorcode.subcommands.vectorise.hash_file", return_value="test_hash"
            ),
            patch("vectorcode.common.try_server", return_value=True),
        ):
            mock_config = Config(project_root=temp_dir)
            mock_get_project_config.return_value = mock_config

            mock_collection = AsyncMock()
            mock_collection.get.return_value = {"ids": [], "metadatas": []}
            mock_get_collection.return_value = mock_collection
            mock_client.get_max_batch_size.return_value = 100

            await vectorise_files(paths=[file1, excluded_file], project_root=temp_dir)

            assert mock_chunked_add.call_count == 1
            call_args = [call[0][0] for call in mock_chunked_add.call_args_list]
            assert excluded_file not in call_args


@pytest.mark.asyncio
async def test_mcp_server():
    mock_client = AsyncMock()
    with (
        patch(
            "vectorcode.mcp_main.find_project_config_dir"
        ) as mock_find_project_config_dir,
        patch("vectorcode.mcp_main.load_config_file") as mock_load_config_file,
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
        patch("mcp.server.fastmcp.FastMCP.add_tool") as mock_add_tool,
        patch("vectorcode.common.try_server", return_value=True),
        patch(
            "vectorcode.mcp_main.ClientManager._create_client", return_value=mock_client
        ),
    ):
        mock_find_project_config_dir.return_value = "/path/to/config"
        mock_load_config_file.return_value = Config(project_root="/path/to/project")

        mock_collection = AsyncMock()
        mock_get_collection.return_value = mock_collection

        await mcp_server()

        assert mock_add_tool.call_count == 5


@pytest.mark.asyncio
async def test_mcp_server_ls_on_start():
    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch(
            "vectorcode.mcp_main.find_project_config_dir"
        ) as mock_find_project_config_dir,
        patch("vectorcode.mcp_main.load_config_file") as mock_load_config_file,
        patch("vectorcode.mcp_main.get_collection") as mock_get_collection,
        patch(
            "vectorcode.mcp_main.get_collections", spec=AsyncMock
        ) as mock_get_collections,
        patch("mcp.server.fastmcp.FastMCP.add_tool") as mock_add_tool,
        patch("vectorcode.common.try_server", return_value=True),
        patch(
            "vectorcode.mcp_main.ClientManager._create_client", return_value=mock_client
        ),
    ):
        from vectorcode.mcp_main import mcp_config

        mcp_config.ls_on_start = True
        mock_find_project_config_dir.return_value = "/path/to/config"
        mock_load_config_file.return_value = Config(project_root="/path/to/project")

        mock_collection.metadata = {"path": "/path/to/project"}
        mock_get_collection.return_value = mock_collection

        async def new_get_collections(clients):
            yield mock_collection

        mock_get_collections.side_effect = new_get_collections

        await mcp_server()

        assert mock_add_tool.call_count == 5
        mock_get_collections.assert_called()


@pytest.mark.asyncio
async def test_ls_files_success():
    ClientManager().clear()
    mock_client = MagicMock()
    mock_collection = MagicMock()
    expected_files = ["/test/project/file1.py", "/test/project/dir/file2.txt"]

    with (
        patch("vectorcode.mcp_main.get_project_config") as mock_get_project_config,
        patch(
            "vectorcode.mcp_main.ClientManager._create_client", return_value=mock_client
        ),
        patch("vectorcode.common.try_server", return_value=True),
        patch("vectorcode.mcp_main.get_collection", return_value=mock_collection),
        patch(
            "vectorcode.mcp_main.list_collection_files", return_value=expected_files
        ) as mock_list_collection_files,
        patch(
            "vectorcode.cli_utils.expand_path", side_effect=lambda x, y: x
        ),  # Mock expand_path to return input
    ):
        mock_get_project_config.return_value = Config(project_root="/test/project")
        result = await ls_files(project_root="/test/project")

        assert result == expected_files
        mock_get_project_config.assert_called_once_with("/test/project")

        mock_list_collection_files.assert_called_once_with(mock_collection)


@pytest.mark.asyncio
async def test_rm_files_success():
    ClientManager().clear()
    mock_client = MagicMock()
    mock_collection = MagicMock()
    files_to_remove = ["/test/project/file1.py", "/test/project/file2.txt"]

    with (
        patch("os.path.isfile", side_effect=lambda x: x in files_to_remove),
        patch("vectorcode.mcp_main.get_project_config") as mock_get_project_config,
        patch(
            "vectorcode.mcp_main.ClientManager._create_client", return_value=mock_client
        ),
        patch("vectorcode.common.try_server", return_value=True),
        patch("vectorcode.mcp_main.get_collection", return_value=mock_collection),
        patch("vectorcode.cli_utils.expand_path", side_effect=lambda x, y: x),
    ):
        mock_get_project_config.return_value = Config(project_root="/test/project")
        mock_collection.delete = AsyncMock()

        await rm_files(files=files_to_remove, project_root="/test/project")

        mock_get_project_config.assert_called_once_with("/test/project")
        mock_collection.delete.assert_called_once_with(
            where={"path": {"$in": files_to_remove}}
        )


def test_arg_parser():
    assert isinstance(get_arg_parser(), ArgumentParser)


def test_args_parsing():
    args = ["--number", "15", "--ls-on-start"]
    parsed = parse_cli_args(args)
    assert parsed.n_results == 15
    assert parsed.ls_on_start
