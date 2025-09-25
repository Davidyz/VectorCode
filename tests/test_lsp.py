import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from lsprotocol.types import WorkspaceFolder
from pygls.exceptions import JsonRpcInternalError, JsonRpcInvalidRequest
from pygls.server import LanguageServer

from vectorcode import __version__
from vectorcode.cli_utils import CliAction, Config, FilesAction, QueryInclude
from vectorcode.database.types import (
    CollectionContent as FileList,
)
from vectorcode.database.types import (
    CollectionInfo as Collection,
)
from vectorcode.database.types import (
    FileInCollection as File,
)
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
    ls.workspace = MagicMock()
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
        patch("vectorcode.lsp_main.get_database_connector"),
        patch(
            "vectorcode.lsp_main.get_reranked_results", new_callable=AsyncMock
        ) as mock_get_reranked_results,
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", MagicMock()) as mock_open,
    ):
        mock_parse_cli_args.return_value = mock_config
        mock_get_reranked_results.return_value = []

        # Configure the MagicMock object to return a string when read() is called
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "{}"  # Return valid JSON
        mock_open.return_value = mock_file

        # Ensure parsed_args.project_root is not None
        mock_config.project_root = "/test/project"
        mock_config.query = ["test"]

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        result = await execute_command(mock_language_server, ["query", "test"])

        assert isinstance(result, list)
        mock_language_server.progress.begin.assert_called()
        mock_language_server.progress.end.assert_called()


@pytest.mark.asyncio
async def test_execute_command_query_invalid_include(mock_language_server, mock_config):
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch("vectorcode.lsp_main.get_database_connector"),
        patch(
            "vectorcode.lsp_main.get_reranked_results", new_callable=AsyncMock
        ) as mock_get_reranked_results,
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", MagicMock()) as mock_open,
    ):
        mock_parse_cli_args.return_value = mock_config
        mock_get_reranked_results.return_value = []

        # Configure the MagicMock object to return a string when read() is called
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "{}"  # Return valid JSON
        mock_open.return_value = mock_file

        # Ensure parsed_args.project_root is not None
        mock_config.project_root = "/test/project"
        mock_config.query = ["test"]
        mock_config.include = [QueryInclude.chunk, QueryInclude.document]

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        result = await execute_command(mock_language_server, ["query", "test"])

        assert result == []
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
        patch("vectorcode.lsp_main.get_database_connector"),
        patch(
            "vectorcode.lsp_main.get_reranked_results", new_callable=AsyncMock
        ) as mock_get_reranked_results,
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", MagicMock()) as mock_open,
    ):
        global DEFAULT_PROJECT_ROOT
        mock_config.project_root = None
        mock_parse_cli_args.return_value = mock_config
        mock_get_reranked_results.return_value = []

        # Configure the MagicMock object to return a string when read() is called
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "{}"  # Return valid JSON
        mock_open.return_value = mock_file

        # Ensure parsed_args.project_root is not None
        DEFAULT_PROJECT_ROOT = "/test/project"
        mock_config.query = ["test"]

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        result = await execute_command(mock_language_server, ["query", "test"])

        assert isinstance(result, list)
        mock_language_server.progress.begin.assert_called()
        mock_language_server.progress.end.assert_called()


@pytest.mark.asyncio
async def test_execute_command_query_workspace_dir(mock_language_server, mock_config):
    workspace_folder = WorkspaceFolder(uri="file:///dummy_dir", name="dummy_dir")
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch("vectorcode.lsp_main.get_database_connector"),
        patch(
            "vectorcode.lsp_main.get_reranked_results", new_callable=AsyncMock
        ) as mock_get_reranked_results,
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("builtins.open", MagicMock()) as mock_open,
    ):
        mock_language_server.workspace = MagicMock()
        mock_language_server.workspace.folders = {"dummy_dir": workspace_folder}
        mock_config.project_root = None
        mock_config.query = ["test"]
        mock_parse_cli_args.return_value = mock_config
        mock_get_reranked_results.return_value = []

        # Configure the MagicMock object to return a string when read() is called
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "{}"  # Return valid JSON
        mock_open.return_value = mock_file

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        result = await execute_command(mock_language_server, ["query", "test"])

        assert isinstance(result, list)
        mock_language_server.progress.begin.assert_called()
        mock_language_server.progress.end.assert_called()
        assert mock_config.project_root == "/dummy_dir"


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
        patch(
            "vectorcode.lsp_main.get_database_connector"
        ) as mock_get_database_connector,
    ):
        mock_parse_cli_args.return_value = mock_config
        mock_db_connector = AsyncMock()
        mock_db_connector.list_collections.return_value = [
            Collection(
                id="dummy",
                path="/test/project",
                embedding_function="",
                database_backend="",
            )
        ]
        mock_get_database_connector.return_value = mock_db_connector

        # Ensure parsed_args.project_root is not None
        mock_config.project_root = "/test/project"

        # Mock the merge_from method
        mock_config.merge_from = AsyncMock(return_value=mock_config)

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

    # Files after expand_globs
    dummy_expanded_files = ["/test/project/file_a.py", "/test/project/file_b.txt"]

    # Mock dependencies
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch(
            "vectorcode.lsp_main.get_database_connector"
        ) as mock_get_database_connector,
        patch(
            "vectorcode.lsp_main.expand_globs", new_callable=AsyncMock
        ) as mock_expand_globs,
        patch("os.path.isfile", lambda x: x in dummy_expanded_files),
        patch("vectorcode.lsp_main.find_exclude_specs", return_value=[]),
        patch("os.cpu_count", return_value=1),
        patch("vectorcode.lsp_main.get_project_config", return_value=mock_config),
    ):
        # Set return values for mocks
        mock_parse_cli_args.return_value = mock_config
        mock_db_connector = AsyncMock()
        mock_get_database_connector.return_value = mock_db_connector

        mock_expand_globs.return_value = (
            dummy_expanded_files  # What expand_globs should return
        )

        mock_config.merge_from = AsyncMock(return_value=mock_config)

        await execute_command(
            mock_language_server,
            ["vectorise", "/test/project", "file_a.py", "file_b.txt"],
        )
        assert mock_db_connector.vectorise.await_args_list == [
            call(file_path="/test/project/file_a.py"),
            call(file_path="/test/project/file_b.txt"),
        ]


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
    ):
        mock_parse_cli_args.return_value = mock_config

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


@pytest.mark.asyncio
async def test_execute_command_files_ls(mock_language_server, mock_config: Config):
    mock_config.action = CliAction.files
    mock_config.files_action = FilesAction.ls
    mock_config.project_root = "/test/project"

    dummy_files = FileList(
        files=[
            File(path="/test/project/file1.py", sha256="1"),
            File(path="/test/project/file2.txt", sha256="2"),
        ]
    )
    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch(
            "vectorcode.lsp_main.get_database_connector"
        ) as mock_get_database_connector,
    ):
        mock_parse_cli_args.return_value = mock_config
        mock_db_connector = AsyncMock()
        mock_db_connector.list_collection_content.return_value = dummy_files
        mock_get_database_connector.return_value = mock_db_connector
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        await execute_command(mock_language_server, ["files", "ls"])
        mock_db_connector.list_collection_content.assert_called_once()


@pytest.mark.asyncio
async def test_execute_command_files_rm(mock_language_server, mock_config: Config):
    mock_config.action = CliAction.files
    mock_config.files_action = FilesAction.rm
    mock_config.project_root = "/test/project"
    mock_config.rm_paths = ["file_to_remove.py", "another_file.txt"]

    expanded_paths = [
        "/test/project/file_to_remove.py",
        "/test/project/another_file.txt",
    ]

    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch(
            "vectorcode.lsp_main.get_database_connector"
        ) as mock_get_database_connector,
        patch(
            "os.path.isfile",
            side_effect=lambda x: x in expanded_paths or x in mock_config.rm_paths,
        ),
        patch(
            "vectorcode.lsp_main.expand_path",
            side_effect=lambda p, *args: os.path.join(mock_config.project_root, p),
        ),
    ):
        mock_parse_cli_args.return_value = mock_config
        mock_db_connector = AsyncMock()
        mock_get_database_connector.return_value = mock_db_connector
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        await execute_command(
            mock_language_server,
            ["files", "rm", "file_to_remove.py", "another_file.txt"],
        )
        mock_db_connector.delete.assert_called_once()


@pytest.mark.asyncio
async def test_execute_command_files_rm_no_files_to_remove(
    mock_language_server, mock_config: Config
):
    mock_config.action = CliAction.files
    mock_config.files_action = FilesAction.rm
    mock_config.project_root = "/test/project"
    mock_config.rm_paths = ["non_existent_file.py"]

    with (
        patch(
            "vectorcode.lsp_main.parse_cli_args", new_callable=AsyncMock
        ) as mock_parse_cli_args,
        patch(
            "vectorcode.lsp_main.get_database_connector"
        ) as mock_get_database_connector,
        patch("os.path.isfile", return_value=False),
        patch(
            "vectorcode.lsp_main.expand_path",
            side_effect=lambda p, *args: os.path.join(mock_config.project_root, p),
        ),
    ):
        mock_parse_cli_args.return_value = mock_config
        mock_db_connector = AsyncMock()
        mock_get_database_connector.return_value = mock_db_connector
        mock_config.merge_from = AsyncMock(return_value=mock_config)

        await execute_command(
            mock_language_server, ["files", "rm", "non_existent_file.py"]
        )
        mock_db_connector.assert_not_called()
