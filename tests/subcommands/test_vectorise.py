import asyncio
import hashlib
import json
import os
import socket
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pathspec
import pytest
from chromadb.api.models.AsyncCollection import AsyncCollection

from vectorcode.cli_utils import Config
from vectorcode.subcommands.vectorise import (
    chunked_add,
    exclude_paths_by_spec,
    get_uuid,
    hash_str,
    include_paths_by_spec,
    load_files_from_include,
    show_stats,
    vectorise,
)


def test_hash_str():
    test_string = "test_string"
    expected_hash = hashlib.sha256(test_string.encode()).hexdigest()
    assert hash_str(test_string) == expected_hash


def test_get_uuid():
    uuid_str = get_uuid()
    assert isinstance(uuid_str, str)
    assert len(uuid_str) == 32  # UUID4 hex string length


@pytest.mark.asyncio
async def test_chunked_add():
    file_path = "test_file.py"
    collection = AsyncMock()
    collection_lock = asyncio.Lock()
    stats = {"add": 0, "update": 0}
    stats_lock = asyncio.Lock()
    configs = Config(chunk_size=100, overlap_ratio=0.2, project_root=".")
    max_batch_size = 50
    semaphore = asyncio.Semaphore(1)

    with patch("vectorcode.chunking.TreeSitterChunker.chunk") as mock_chunk:
        mock_chunk.return_value = ["chunk1", "chunk2"]
        await chunked_add(
            file_path,
            collection,
            collection_lock,
            stats,
            stats_lock,
            configs,
            max_batch_size,
            semaphore,
        )

    assert stats["add"] == 1
    assert stats["update"] == 0
    collection.add.assert_called()
    assert collection.add.call_count == 1


@patch("tabulate.tabulate")
def test_show_stats_pipe_false(mock_tabulate, capsys):
    configs = Config(pipe=False)
    stats = {"add": 1, "update": 2, "removed": 3}
    show_stats(configs, stats)
    mock_tabulate.assert_called_once()


def test_show_stats_pipe_true(capsys):
    configs = Config(pipe=True)
    stats = {"add": 1, "update": 2, "removed": 3}
    show_stats(configs, stats)
    captured = capsys.readouterr()
    assert captured.out == json.dumps(stats) + "\n"


def test_exclude_paths_by_spec():
    paths = ["file1.py", "file2.py", "exclude.py"]
    specs = pathspec.PathSpec.from_lines(
        pattern_factory="gitwildmatch", lines=["exclude.py"]
    )
    excluded_paths = exclude_paths_by_spec(paths, specs)
    assert "exclude.py" not in excluded_paths
    assert len(excluded_paths) == 2


def test_include_paths_by_spec():
    paths = ["file1.py", "file2.py", "include.py"]
    specs = pathspec.PathSpec.from_lines(
        pattern_factory="gitwildmatch", lines=["include.py", "file1.py"]
    )
    included_paths = include_paths_by_spec(paths, specs)
    assert "file2.py" not in included_paths
    assert len(included_paths) == 2


@patch("os.path.isfile", return_value=True)
@patch("os.path.join", return_value="path/to/.vectorcode/vectorcode.include")
def test_load_files_from_include(mock_join, mock_isfile, tmpdir):
    include_file_content = "file1.py\nfile2.py"
    include_file_path = tmpdir.join("vectorcode.include")
    include_file_path.write(include_file_content)
    project_root = str(tmpdir)

    with patch("builtins.open", return_value=open(str(include_file_path), "r")):
        with patch("pathspec.PathSpec.check_tree_files") as mock_check_tree_files:
            mock_check_tree_files.return_value = [
                MagicMock(file="file1.py", include=True),
                MagicMock(file="file2.py", include=True),
                MagicMock(file="file3.py", include=False),
            ]
            files = load_files_from_include(project_root)
            assert "file3.py" not in files
            assert len(files) == 2


@pytest.mark.asyncio
async def test_vectorise(capsys):
    configs = Config(
        host="test_host",
        port=1234,
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        files=["test_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )
    mock_client = AsyncMock()
    mock_collection = MagicMock(spec=AsyncCollection)
    mock_collection.get.return_value = {"ids": []}
    mock_collection.delete.return_value = None
    mock_collection.metadata = {
        "embedding_function": "SentenceTransformerEmbeddingFunction",
        "path": "/test_project",
        "hostname": socket.gethostname(),
        "created-by": "VectorCode",
        "username": os.environ.get("USER", os.environ.get("USERNAME", "DEFAULT_USER")),
    }
    mock_client.get_max_batch_size.return_value = 50
    mock_embedding_function = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "vectorcode.subcommands.vectorise.get_client", return_value=mock_client
            )
        )
        stack.enter_context(patch("os.path.isfile", return_value=False))
        stack.enter_context(
            patch(
                "vectorcode.subcommands.vectorise.expand_globs",
                return_value=["test_file.py"],
            )
        )
        mock_chunked_add = stack.enter_context(
            patch("vectorcode.subcommands.vectorise.chunked_add", return_value=None)
        )
        stack.enter_context(
            patch(
                "vectorcode.common.get_embedding_function",
                return_value=mock_embedding_function,
            )
        )
        stack.enter_context(
            patch(
                "vectorcode.subcommands.vectorise.get_collection",
                return_value=mock_collection,
            )
        )

        result = await vectorise(configs)
        assert result == 0
        assert mock_chunked_add.call_count == 1


@pytest.mark.asyncio
async def test_vectorise_cancelled():
    configs = Config(
        host="test_host",
        port=1234,
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        files=["test_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )

    async def mock_chunked_add(*args, **kwargs):
        raise asyncio.CancelledError

    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch(
            "vectorcode.subcommands.vectorise.chunked_add", side_effect=mock_chunked_add
        ) as mock_add,
        patch("sys.stderr") as mock_stderr,
        patch("vectorcode.subcommands.vectorise.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.vectorise.get_collection",
            return_value=mock_collection,
        ),
        patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
        patch(
            "os.path.isfile",
            lambda x: not (x.endswith("gitignore") or x.endswith("vectorcode.exclude")),
        ),
    ):
        result = await vectorise(configs)
        assert result == 1
        mock_add.assert_called_once()
        mock_stderr.write.assert_called()


@pytest.mark.asyncio
async def test_vectorise_orphaned_files():
    configs = Config(
        host="test_host",
        port=1234,
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        files=["test_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )

    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    # Define a mock response for collection.get in vectorise
    get_return = {
        "metadatas": [{"path": "test_file.py"}, {"path": "non_existent_file.py"}]
    }
    mock_collection.get.side_effect = [
        {"ids": []},  # Return value for chunked_add
        get_return,  # Return value for orphaned files
    ]
    mock_collection.delete.return_value = None

    # Mock TreeSitterChunker
    mock_chunker = AsyncMock()

    def chunk(*args, **kwargs):
        return ["chunk1", "chunk2"]

    mock_chunker.chunk = chunk

    # Mock os.path.isfile
    def is_file_side_effect(path):
        if path == "non_existent_file.py":
            return False
        elif path.endswith(".gitignore") or path.endswith("vectorcode.exclude"):
            return False
        else:
            return True

    with (
        patch("os.path.isfile", side_effect=is_file_side_effect),
        patch(
            "vectorcode.subcommands.vectorise.TreeSitterChunker",
            return_value=mock_chunker,
        ),
        patch("vectorcode.subcommands.vectorise.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.vectorise.get_collection",
            return_value=mock_collection,
        ),
        patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
        patch(
            "vectorcode.subcommands.vectorise.expand_globs",
            return_value=["test_file.py"],  # Ensure expand_globs returns a valid file
        ),
    ):
        result = await vectorise(configs)

        assert result == 0
        mock_collection.delete.assert_called_once_with(
            where={"path": {"$in": ["non_existent_file.py"]}}
        )


@pytest.mark.asyncio
async def test_vectorise_collection_index_error():
    configs = Config(
        host="test_host",
        port=1234,
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        files=["test_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )

    mock_client = AsyncMock()

    with (
        patch("vectorcode.subcommands.vectorise.get_client", return_value=mock_client),
        patch("vectorcode.subcommands.vectorise.get_collection") as mock_get_collection,
        patch("os.path.isfile", return_value=False),
    ):
        mock_get_collection.side_effect = IndexError("Collection not found")
        result = await vectorise(configs)
        assert result == 1


@pytest.mark.asyncio
async def test_vectorise_verify_ef_false():
    configs = Config(
        host="test_host",
        port=1234,
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        files=["test_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )
    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.vectorise.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.vectorise.get_collection",
            return_value=mock_collection,
        ),
        patch("vectorcode.subcommands.vectorise.verify_ef", return_value=False),
        patch("os.path.isfile", return_value=False),
    ):
        result = await vectorise(configs)
        assert result == 1


@pytest.mark.asyncio
async def test_vectorise_gitignore():
    configs = Config(
        host="test_host",
        port=1234,
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        files=["test_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )
    mock_client = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {"metadatas": []}

    with (
        patch("vectorcode.subcommands.vectorise.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.vectorise.get_collection",
            return_value=mock_collection,
        ),
        patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
        patch(
            "os.path.isfile",
            side_effect=lambda path: path
            == os.path.join("/test_project", ".gitignore"),
        ),
        patch("builtins.open", return_value=MagicMock()),
        patch(
            "vectorcode.subcommands.vectorise.expand_globs",
            return_value=["test_file.py"],
        ),
        patch(
            "vectorcode.subcommands.vectorise.exclude_paths_by_spec"
        ) as mock_exclude_paths,
    ):
        await vectorise(configs)
        mock_exclude_paths.assert_called_once()


@pytest.mark.asyncio
async def test_vectorise_exclude_file(tmpdir):
    # Create a temporary .vectorcode directory and vectorcode.exclude file
    exclude_dir = tmpdir.mkdir(".vectorcode")
    exclude_file = exclude_dir.join("vectorcode.exclude")
    exclude_file.write("excluded_file.py\n")

    configs = Config(
        host="test_host",
        port=1234,
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root=str(tmpdir),
        files=["test_file.py", "excluded_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )
    mock_client = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {"ids": []}

    with (
        patch("vectorcode.subcommands.vectorise.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.vectorise.get_collection",
            return_value=mock_collection,
        ),
        patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
        patch(
            "os.path.isfile",
            side_effect=lambda path: True if path == str(exclude_file) else False,
        ),
        patch("builtins.open", return_value=open(str(exclude_file), "r")),
        patch(
            "vectorcode.subcommands.vectorise.expand_globs",
            return_value=["test_file.py", "excluded_file.py"],
        ),
        patch("vectorcode.subcommands.vectorise.chunked_add") as mock_chunked_add,
    ):
        await vectorise(configs)
        # Assert that chunked_add is only called for test_file.py, not excluded_file.py
        call_args = [call[0][0] for call in mock_chunked_add.call_args_list]
        assert "excluded_file.py" not in call_args
        assert "test_file.py" in call_args
        assert mock_chunked_add.call_count == 1
