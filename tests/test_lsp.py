from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pygls.exceptions import JsonRpcInternalError, JsonRpcInvalidRequest
from pygls.server import LanguageServer

from vectorcode import __version__
from vectorcode.cli_utils import CliAction, Config, QueryInclude
from vectorcode.lsp_main import (
    execute_command,
    lsp_start,
)


@pytest.fixture
def mock_language_server():
    ls = MagicMock(spec=LanguageServer)
    ls.progress.create_async = AsyncMock()
    ls.progress.begin = MagicMock()
    ls.progress.end = MagicMock()
    return ls


@pytest.fixture
def mock_config():
    # config = MagicMock(spec=Config)
    config = Config()
    config.host = "localhost"
    config.port = 8000
    config.action = CliAction.query
    config.project_root = "/test/project"
    config.use_absolute_path = True
    config.pipe = False
    config.overlap_ratio = 0.2
    config.query_exclude = []
    config.include = [QueryInclude.path]
    config.query_multipler = 10
    return config


@pytest.mark.asyncio
async def test_execute_command_query(mock_language_server, mock_config):
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch("vectorcode.lsp_main.ClientManager"),
        patch("vectorcode.lsp_main.get_collection", new_callable=AsyncMock),
        patch(
            "vectorcode.lsp_main.build_query_results", new_callable=AsyncMock
        ) as mock_get_query_result_files,
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", MagicMock()) as mock_open,
    ):
        mock_parse_cli_args.return_value = mock_config
        mock_get_query_result_files.return_value = ["/test/file.txt"]

        # Configure the MagicMock object to return a string when read() is called
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "{}"  # Return valid JSON
        mock_open.return_value = mock_file

        # Ensure parsed_args.project_root is not None
        mock_config.project_root = "/test/project"

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        result = await execute_command(mock_language_server, ["query", "test"])

        assert isinstance(result, list)
        mock_language_server.progress.begin.assert_called()
        mock_language_server.progress.end.assert_called()


@pytest.mark.asyncio
async def test_execute_command_query_default_proj_root(
    mock_language_server, mock_config
):
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch("vectorcode.lsp_main.ClientManager"),
        patch("vectorcode.lsp_main.get_collection", new_callable=AsyncMock),
        patch(
            "vectorcode.lsp_main.build_query_results", new_callable=AsyncMock
        ) as mock_get_query_result_files,
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", MagicMock()) as mock_open,
    ):
        global DEFAULT_PROJECT_ROOT

        mock_config.project_root = None
        mock_parse_cli_args.return_value = mock_config
        mock_get_query_result_files.return_value = ["/test/file.txt"]

        # Configure the MagicMock object to return a string when read() is called
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "{}"  # Return valid JSON
        mock_open.return_value = mock_file

        # Ensure parsed_args.project_root is not None
        DEFAULT_PROJECT_ROOT = "/test/project"

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        result = await execute_command(mock_language_server, ["query", "test"])

        assert isinstance(result, list)
        mock_language_server.progress.begin.assert_called()
        mock_language_server.progress.end.assert_called()


@pytest.mark.asyncio
async def test_execute_command_ls(mock_language_server, mock_config):
    mock_config.action = CliAction.ls
    mock_config.embedding_function = "SentenceTransformerEmbeddingFunction"
    mock_config.embedding_params = {}
    mock_config.db_settings = {}
    mock_config.hnsw = None  # Add the hnsw attribute

    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch("vectorcode.lsp_main.ClientManager"),
        patch(
            "vectorcode.lsp_main.get_collection_list", new_callable=AsyncMock
        ) as mock_get_collection_list,
        patch("vectorcode.common.get_embedding_function") as mock_embedding_function,
        patch("vectorcode.common.get_collection") as mock_get_collection,
    ):
        mock_parse_cli_args.return_value = mock_config

        # Ensure parsed_args.project_root is not None
        mock_config.project_root = "/test/project"

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        mock_get_collection_list.return_value = [{"project": "/test/project"}]
        mock_embedding_function.return_value = MagicMock()  # Mock embedding function
        mock_get_collection.return_value = MagicMock()

        result = await execute_command(mock_language_server, ["ls"])

        assert isinstance(result, list)
        mock_language_server.progress.begin.assert_called()
        mock_language_server.progress.end.assert_called()


@pytest.mark.asyncio
async def test_execute_command_vectorise(mock_language_server, mock_config: Config):
    mock_config.action = CliAction.vectorise  # Set action to vectorise
    mock_config.project_root = "/test/project"  # Ensure project_root is set
    mock_config.files = None  # Simulate no files explicitly passed, so load_files_from_include is called
    mock_config.recursive = True
    mock_config.include_hidden = False
    mock_config.force = False  # To test exclude_paths_by_spec path

    # Files that load_files_from_include will return and expand_globs will process
    dummy_initial_files = ["file_a.py", "file_b.txt"]
    # Files after expand_globs
    dummy_expanded_files = ["/test/project/file_a.py", "/test/project/file_b.txt"]

    # Mock dependencies
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch("vectorcode.lsp_main.ClientManager") as MockClientManager,
        patch(
            "vectorcode.lsp_main.get_collection", new_callable=AsyncMock
        ) as mock_get_collection,
        patch(
            "vectorcode.lsp_main.expand_globs", new_callable=AsyncMock
        ) as mock_expand_globs,
        patch(
            "vectorcode.lsp_main.find_exclude_specs", return_value=[]
        ) as mock_find_exclude_specs,
        patch(
            "vectorcode.lsp_main.exclude_paths_by_spec",
            side_effect=lambda files, spec: files,
        ) as mock_exclude_paths_by_spec,
        patch(
            "vectorcode.lsp_main.chunked_add", new_callable=AsyncMock
        ) as mock_chunked_add,
        patch(
            "vectorcode.lsp_main.load_files_from_include",
            return_value=dummy_initial_files,
        ) as mock_load_files_from_include,
        patch("os.cpu_count", return_value=1),  # For asyncio.Semaphore
        patch(
            "vectorcode.lsp_main.remove_orphanes", new_callable=AsyncMock
        ) as mock_remove_orphanes,
    ):
        from unittest.mock import ANY

        from lsprotocol import types

        @asynccontextmanager
        async def _get_client(*args):
            yield mock_client

        # Set return values for mocks
        mock_parse_cli_args.return_value = mock_config
        mock_client = AsyncMock()
        MockClientManager.return_value.get_client.side_effect = _get_client
        mock_collection = AsyncMock()
        mock_get_collection.return_value = mock_collection
        mock_client.get_max_batch_size.return_value = 100  # Mock batch size

        mock_expand_globs.return_value = (
            dummy_expanded_files  # What expand_globs should return
        )

        # Mock merge_from as it's called
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        # Execute the command
        result = await execute_command(
            mock_language_server, ["vectorise", "/test/project"]
        )
        assert isinstance(result, dict) and all(
            k in ("add", "update", "removed", "failed", "skipped")
            for k in result.keys()
        )

        # Assertions
        mock_language_server.progress.create_async.assert_called_once()
        mock_language_server.progress.begin.assert_called_once_with(
            ANY,  # progress_token
            types.WorkDoneProgressBegin(
                title="VectorCode", message="Vectorising files...", percentage=0
            ),
        )

        mock_load_files_from_include.assert_called_once_with(
            str(mock_config.project_root)
        )
        mock_expand_globs.assert_called_once_with(
            dummy_initial_files,  # Should be the result of load_files_from_include
            recursive=mock_config.recursive,
            include_hidden=mock_config.include_hidden,
        )
        mock_find_exclude_specs.assert_called_once()
        mock_exclude_paths_by_spec.assert_not_called()  # Because mock_find_exclude_specs returns empty list (no specs to exclude by)
        mock_client.get_max_batch_size.assert_called_once()

        # Check chunked_add calls
        assert mock_chunked_add.call_count == len(dummy_expanded_files)
        for file_path in dummy_expanded_files:
            mock_chunked_add.assert_any_call(
                file_path,
                mock_collection,
                ANY,  # asyncio.Lock object
                ANY,  # stats dict
                ANY,  # stats_lock
                ANY,
                100,  # max_batch_size
                ANY,  # semaphore
            )
        # Check progress report calls
        assert mock_language_server.progress.report.call_count == len(
            dummy_expanded_files
        )
        mock_remove_orphanes.assert_called_once()
        mock_language_server.progress.end.assert_called_once()


@pytest.mark.asyncio
async def test_execute_command_unsupported_action(
    mock_language_server, mock_config, capsys
):
    mock_config.action = "invalid_action"
    mock_config.project_root = "/test/project"  # Add project_root
    mock_config.embedding_function = "SentenceTransformerEmbeddingFunction"
    mock_config.embedding_params = {}
    mock_config.db_settings = {}
    mock_config.hnsw = None

    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch(
            "vectorcode.lsp_main.get_collection", new_callable=AsyncMock
        ) as mock_get_collection,
    ):
        mock_parse_cli_args.return_value = mock_config

        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        with pytest.raises((JsonRpcInternalError, JsonRpcInvalidRequest)):
            await execute_command(mock_language_server, ["invalid_action"])


@pytest.mark.asyncio
async def test_lsp_start_version(capsys):
    with patch("sys.argv", ["lsp_main.py", "--version"]):
        result = await lsp_start()
        captured = capsys.readouterr()
        assert __version__ in captured.out
        assert result == 0


@pytest.mark.asyncio
async def test_lsp_start_no_project_root():
    with patch("sys.argv", ["lsp_main.py"]):
        with (
            patch("vectorcode.lsp_main.find_project_root") as mock_find_project_root,
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            mock_find_project_root.return_value = "/test/project"
            await lsp_start()
            mock_to_thread.assert_called_once()
            from vectorcode.lsp_main import (
                DEFAULT_PROJECT_ROOT,
            )

            assert DEFAULT_PROJECT_ROOT == "/test/project"


@pytest.mark.asyncio
async def test_lsp_start_with_project_root():
    with patch("sys.argv", ["lsp_main.py", "--project_root", "/test/project"]):
        with patch("asyncio.to_thread") as mock_to_thread:
            await lsp_start()
            mock_to_thread.assert_called_once()
            from vectorcode.lsp_main import (
                DEFAULT_PROJECT_ROOT,
            )

            assert DEFAULT_PROJECT_ROOT == "/test/project"


@pytest.mark.asyncio
async def test_lsp_start_find_project_root_none():
    with patch("sys.argv", ["lsp_main.py"]):
        with (
            patch("vectorcode.lsp_main.find_project_root") as mock_find_project_root,
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            mock_find_project_root.return_value = None
            await lsp_start()
            mock_to_thread.assert_called_once()
            from vectorcode.lsp_main import (
                DEFAULT_PROJECT_ROOT,
            )

            assert DEFAULT_PROJECT_ROOT is None


@pytest.mark.asyncio
async def test_execute_command_no_default_project_root(
    mock_language_server, mock_config
):
    global DEFAULT_PROJECT_ROOT
    DEFAULT_PROJECT_ROOT = None
    mock_config.project_root = None
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
    ):
        mock_parse_cli_args.return_value = mock_config
        with pytest.raises((AssertionError, JsonRpcInternalError)):
            await execute_command(mock_language_server, ["query", "test"])
    DEFAULT_PROJECT_ROOT = None  # Reset the global variable
