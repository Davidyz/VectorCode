import asyncio
import hashlib
import os
import socket
import tempfile
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import numpy
import pytest
from chromadb.api.models.AsyncCollection import AsyncCollection
from tree_sitter import Point

from vectorcode.chunking import Chunk
from vectorcode.cli_utils import CliAction, Config
from vectorcode.subcommands.vectorise import (
    VectoriseStats,
    chunked_add,
    exclude_paths_by_spec,
    find_exclude_specs,
    get_uuid,
    hash_file,
    hash_str,
    load_files_from_include,
    show_stats,
    vectorise,
)


def test_hash_str():
    test_string = "test_string"
    expected_hash = hashlib.sha256(test_string.encode()).hexdigest()
    assert hash_str(test_string) == expected_hash


def test_hash_file_basic():
    content = b"This is a test file for hashing."
    expected_hash = hashlib.sha256(content).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        actual_hash = hash_file(tmp_file_path)
        assert actual_hash == expected_hash
    finally:
        os.remove(tmp_file_path)


def test_hash_file_empty():
    content = b""
    expected_hash = hashlib.sha256(content).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        actual_hash = hash_file(tmp_file_path)
        assert actual_hash == expected_hash
    finally:
        os.remove(tmp_file_path)


def test_get_uuid():
    uuid_str = get_uuid()
    assert isinstance(uuid_str, str)
    assert len(uuid_str) == 32  # UUID4 hex string length


@pytest.mark.asyncio
async def test_chunked_add():
    file_path = "test_file.py"
    collection = AsyncMock()
    collection_lock = asyncio.Lock()
    stats = VectoriseStats()
    stats_lock = asyncio.Lock()
    configs = Config(chunk_size=100, overlap_ratio=0.2, project_root=".")
    max_batch_size = 50
    semaphore = asyncio.Semaphore(1)

    with (
        patch("vectorcode.chunking.TreeSitterChunker.chunk") as mock_chunk,
        patch("vectorcode.subcommands.vectorise.hash_file") as mock_hash_file,
    ):
        mock_hash_file.return_value = "hash1"
        mock_chunk.return_value = [Chunk("chunk1", Point(1, 0), Point(1, 5)), "chunk2"]
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

    assert stats.add == 1
    assert stats.update == 0
    collection.add.assert_called()
    assert collection.add.call_count == 1


@pytest.mark.asyncio
async def test_chunked_add_with_existing():
    file_path = "test_file.py"
    collection = AsyncMock()
    collection.get = AsyncMock()
    collection.get.return_value = {"ids": ["id1"], "metadatas": [{"sha256": "hash1"}]}
    collection_lock = asyncio.Lock()
    stats = VectoriseStats()
    stats_lock = asyncio.Lock()
    configs = Config(chunk_size=100, overlap_ratio=0.2, project_root=".")
    max_batch_size = 50
    semaphore = asyncio.Semaphore(1)

    with (
        patch("vectorcode.chunking.TreeSitterChunker.chunk") as mock_chunk,
        patch("vectorcode.subcommands.vectorise.hash_file") as mock_hash_file,
    ):
        mock_hash_file.return_value = "hash1"
        mock_chunk.return_value = [Chunk("chunk1", Point(1, 0), Point(1, 5)), "chunk2"]
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

    assert stats.add == 0
    assert stats.update == 0
    collection.add.assert_not_called()


@pytest.mark.asyncio
async def test_chunked_add_update_existing():
    file_path = "test_file.py"
    collection = AsyncMock()
    collection.get = AsyncMock()
    collection.get.return_value = {"ids": ["id1"], "metadatas": [{"sha256": "hash1"}]}
    collection_lock = asyncio.Lock()
    stats = VectoriseStats()
    stats_lock = asyncio.Lock()
    configs = Config(chunk_size=100, overlap_ratio=0.2, project_root=".")
    max_batch_size = 50
    semaphore = asyncio.Semaphore(1)

    with (
        patch("vectorcode.chunking.TreeSitterChunker.chunk") as mock_chunk,
        patch("vectorcode.subcommands.vectorise.hash_file") as mock_hash_file,
    ):
        mock_hash_file.return_value = "hash2"
        mock_chunk.return_value = [Chunk("chunk1", Point(1, 0), Point(1, 5)), "chunk2"]
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

    assert stats.add == 0
    assert stats.update == 1
    collection.add.assert_called()


@pytest.mark.asyncio
async def test_chunked_add_empty_file():
    file_path = "test_file.py"
    collection = AsyncMock()
    collection_lock = asyncio.Lock()
    stats = VectoriseStats(**{"add": 0, "update": 0})
    stats_lock = asyncio.Lock()
    configs = Config(chunk_size=100, overlap_ratio=0.2, project_root=".")
    max_batch_size = 50
    semaphore = asyncio.Semaphore(1)

    with (
        patch("vectorcode.chunking.TreeSitterChunker.chunk") as mock_chunk,
        patch("vectorcode.subcommands.vectorise.hash_file") as mock_hash_file,
    ):
        mock_hash_file.return_value = "hash1"
        mock_chunk.return_value = []
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

    assert stats.add == 0
    assert stats.update == 0
    assert collection.add.call_count == 0


@patch("tabulate.tabulate")
def test_show_stats_pipe_false(mock_tabulate, capsys):
    configs = Config(pipe=False)
    stats = VectoriseStats(**{"add": 1, "update": 2, "removed": 3})
    show_stats(configs, stats)
    mock_tabulate.assert_called_once()


def test_show_stats_pipe_true(capsys):
    configs = Config(pipe=True)
    stats = VectoriseStats(**{"add": 1, "update": 2, "removed": 3})
    show_stats(configs, stats)
    captured = capsys.readouterr()
    assert captured.out.strip() == (stats.to_json())


def test_exclude_paths_by_spec():
    with tempfile.TemporaryDirectory() as dir:
        paths = list(
            os.path.join(dir, i) for i in ["file1.py", "file2.py", "exclude.py"]
        )
        spec_path = os.path.join(dir, ".gitignore")
        with open(spec_path, mode="w") as spec_file:
            spec_file.writelines(["exclude.py"])

        paths_after_exclude = exclude_paths_by_spec(paths, spec_path)
        assert "exclude.py" not in paths_after_exclude
        assert len(paths_after_exclude) == 2
        os.remove(spec_path)


def test_nested_exclude_paths_by_spec():
    paths = [
        "file1.py",
        "file2.py",
        "exclude.py",
        os.path.join("nested", "nested_exclude.py"),
    ]
    with tempfile.TemporaryDirectory() as project_root:
        paths = [os.path.join(project_root, i) for i in paths]
        with open(os.path.join(project_root, ".gitignore"), mode="w") as fin:
            fin.writelines(["/exclude.py"])

        nested_git_dir = os.path.join(project_root, "nested")
        os.makedirs(nested_git_dir, exist_ok=True)
        with open(os.path.join(nested_git_dir, ".gitignore"), mode="w") as fin:
            fin.writelines(["/nested_exclude.py"])

        specs = find_exclude_specs(Config(project_root=project_root, recursive=True))
        paths_after_exclude = paths[:]
        for spec in specs:
            paths_after_exclude = exclude_paths_by_spec(paths_after_exclude, spec)
        assert "exclude.py" not in paths_after_exclude
        assert "nested/nested_exclude.py" not in paths_after_exclude
        assert len(paths_after_exclude) == 2


@patch("os.path.isfile")
@patch("pathspec.PathSpec.check_tree_files")
def test_load_files_from_local_include(mock_check_tree_files, mock_isfile, tmp_path):
    """Tests loading files when a local '.vectorcode/vectorcode.include' exists."""
    project_root = str(tmp_path)
    local_include_dir = tmp_path / ".vectorcode"
    local_include_dir.mkdir()
    local_include_file = local_include_dir / "vectorcode.include"
    local_include_content = "local_file1.py\nlocal_file2.py"
    local_include_file.write_text(local_include_content)

    # Mock os.path.isfile to return True only for the local file
    mock_isfile.side_effect = lambda p: str(p) == str(local_include_file)

    # Mock check_tree_files
    mock_check_tree_files.return_value = [
        MagicMock(file="local_file1.py", include=True),
        MagicMock(file="local_file2.py", include=True),
        MagicMock(file="ignored_file.py", include=False),
    ]

    # Use mock_open for the specific local file path
    m_open = mock_open(read_data=local_include_content)
    with patch("builtins.open", m_open):
        files = load_files_from_include(project_root)

    assert "local_file1.py" in files
    assert "local_file2.py" in files
    assert "ignored_file.py" not in files
    assert len(files) == 2
    mock_isfile.assert_any_call(str(local_include_file))
    m_open.assert_called_once_with(str(local_include_file))
    mock_check_tree_files.assert_called_once()


@patch("os.path.isfile")
@patch("pathspec.PathSpec.check_tree_files")
def test_load_files_from_global_include(mock_check_tree_files, mock_isfile, tmp_path):
    """Tests loading files when only a global include spec exists."""
    project_root = str(tmp_path)
    local_include_file = tmp_path / ".vectorcode" / "vectorcode.include"

    # Simulate a global include file
    # Note: We don't actually need the real global path, just a path to use in mocks
    temp_global_include_dir = tmp_path / "global_config"
    temp_global_include_dir.mkdir()
    global_include_file = temp_global_include_dir / "vectorcode.include"
    global_include_content = "global_file1.py\nglobal_file2.py"
    global_include_file.write_text(global_include_content)

    # Mock os.path.isfile: False for local, True for (mocked) global
    mock_isfile.side_effect = lambda p: str(p) == str(global_include_file)

    # Mock check_tree_files
    mock_check_tree_files.return_value = [
        MagicMock(file="global_file1.py", include=True),
        MagicMock(file="global_file2.py", include=True),
        MagicMock(file="ignored_global.py", include=False),
    ]

    m_open = mock_open(read_data=global_include_content)
    # Patch builtins.open and the GLOBAL_INCLUDE_SPEC constant used internally
    with (
        patch("builtins.open", m_open),
        patch(
            "vectorcode.subcommands.vectorise.GLOBAL_INCLUDE_SPEC",
            str(global_include_file),
        ),
    ):
        files = load_files_from_include(project_root)

    assert "global_file1.py" in files
    assert "global_file2.py" in files
    assert "ignored_global.py" not in files
    assert len(files) == 2
    mock_isfile.assert_any_call(str(local_include_file))
    mock_isfile.assert_any_call(str(global_include_file))
    m_open.assert_called_once_with(
        str(global_include_file)
    )  # Check the global file was opened
    mock_check_tree_files.assert_called_once()


@patch("os.path.isfile", return_value=False)  # Neither local nor global exists
@patch("pathspec.PathSpec.check_tree_files")
def test_load_files_from_include_no_files(mock_check_tree_files, mock_isfile, tmp_path):
    """Tests behavior when neither local nor global include files exist."""
    project_root = str(tmp_path)
    local_include_file = tmp_path / ".vectorcode" / "vectorcode.include"
    # Assume a mocked global path for the check
    mocked_global_path = "/mock/global/.config/vectorcode/vectorcode.include"

    with patch(
        "vectorcode.subcommands.vectorise.GLOBAL_INCLUDE_SPEC", mocked_global_path
    ):
        files = load_files_from_include(project_root)

    assert files == []
    mock_isfile.assert_any_call(str(local_include_file))
    mock_isfile.assert_any_call(mocked_global_path)
    mock_check_tree_files.assert_not_called()


@pytest.mark.asyncio
async def test_vectorise(capsys):
    configs = Config(
        db_url="http://test_host:1234",
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
            patch("vectorcode.subcommands.vectorise.ClientManager"),
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
        db_url="http://test_host:1234",
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
        patch("vectorcode.subcommands.vectorise.ClientManager") as MockClientManager,
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
        MockClientManager.return_value._create_client.return_value = mock_client
        result = await vectorise(configs)
        assert result == 1
        mock_add.assert_called_once()
        mock_stderr.write.assert_called()


@pytest.mark.asyncio
async def test_vectorise_orphaned_files():
    configs = Config(
        db_url="http://test_host:1234",
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        files=["test_file.py"],
        recursive=False,
        force=False,
        pipe=False,
    )

    AsyncMock()
    mock_collection = AsyncMock()

    # Define a mock response for collection.get in vectorise
    get_return = {
        "metadatas": [{"path": "test_file.py"}, {"path": "non_existent_file.py"}]
    }
    mock_collection.get.side_effect = [
        {"ids": [], "metadatas": []},  # Return value for chunked_add
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

    mock_embedding_function = MagicMock(return_value=numpy.random.random((100,)))
    with (
        patch("os.path.isfile", side_effect=is_file_side_effect),
        patch(
            "vectorcode.subcommands.vectorise.TreeSitterChunker",
            return_value=mock_chunker,
        ),
        patch("vectorcode.subcommands.vectorise.ClientManager"),
        patch(
            "vectorcode.subcommands.vectorise.get_collection",
            return_value=mock_collection,
        ),
        patch(
            "vectorcode.subcommands.vectorise.get_embedding_function",
            return_value=mock_embedding_function,
        ),
        patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
        patch(
            "vectorcode.subcommands.vectorise.expand_globs",
            return_value=["test_file.py"],  # Ensure expand_globs returns a valid file
        ),
        patch("vectorcode.subcommands.vectorise.hash_file") as mock_hash_file,
    ):
        mock_hash_file.return_value = "hash1"
        result = await vectorise(configs)

        assert result == 0
        mock_collection.delete.assert_called_once_with(
            where={"path": {"$in": ["non_existent_file.py"]}}
        )


@pytest.mark.asyncio
async def test_vectorise_collection_index_error():
    configs = Config(
        db_url="http://test_host:1234",
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
        patch("vectorcode.subcommands.vectorise.ClientManager") as MockClientManager,
        patch("vectorcode.subcommands.vectorise.get_collection") as mock_get_collection,
        patch("os.path.isfile", return_value=False),
    ):
        MockClientManager.return_value._create_client.return_value = mock_client
        mock_get_collection.side_effect = IndexError("Collection not found")
        result = await vectorise(configs)
        assert result == 1


@pytest.mark.asyncio
async def test_vectorise_verify_ef_false():
    configs = Config(
        db_url="http://test_host:1234",
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
        patch("vectorcode.subcommands.vectorise.ClientManager") as MockClientManager,
        patch(
            "vectorcode.subcommands.vectorise.get_collection",
            return_value=mock_collection,
        ),
        patch("vectorcode.subcommands.vectorise.verify_ef", return_value=False),
        patch("os.path.isfile", return_value=False),
    ):
        MockClientManager.return_value._create_client.return_value = mock_client
        result = await vectorise(configs)
        assert result == 1


@pytest.mark.asyncio
async def test_vectorise_gitignore():
    configs = Config(
        db_url="http://test_host:1234",
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
        patch("vectorcode.subcommands.vectorise.ClientManager") as MockClientManager,
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
        MockClientManager.return_value._create_client.return_value = mock_client
        await vectorise(configs)
        mock_exclude_paths.assert_called_once()


@pytest.mark.asyncio
async def test_vectorise_exclude_file():
    # Create a temporary .vectorcode directory and vectorcode.exclude file
    with tempfile.TemporaryDirectory() as tmpdir:
        exclude_dir = os.path.join(tmpdir, ".vectorcode")
        nested_dir = os.path.join(tmpdir, "nested")

        os.makedirs(exclude_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)

        exclude_spec = os.path.join(exclude_dir, "vectorcode.exclude")
        with open(exclude_spec, mode="w") as fin:
            fin.writelines(["excluded_file.py"])
        with open(os.path.join(nested_dir, ".gitignore"), "w") as fin:
            fin.writelines(["excluded_file.py"])
        nested_file_path = os.path.join(nested_dir, "nested_excluded_file.py")
        with open(nested_file_path, "w") as fin:
            # non-recursive case. This file should be included.
            fin.writelines(['print("hello world")'])

        configs = Config(
            db_url="http://test_host:1234",
            db_path="test_db",
            embedding_function="SentenceTransformerEmbeddingFunction",
            embedding_params={},
            project_root=str(tmpdir),
            files=[
                os.path.join(tmpdir, "test_file.py"),
                os.path.join(tmpdir, "excluded_file.py"),
                nested_file_path,
            ],
            recursive=False,
            force=False,
            pipe=False,
        )
        mock_client = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.get.return_value = {"ids": []}

        with (
            patch(
                "vectorcode.subcommands.vectorise.ClientManager"
            ) as MockClientManager,
            patch(
                "vectorcode.subcommands.vectorise.get_collection",
                return_value=mock_collection,
            ),
            patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
            patch(
                "vectorcode.subcommands.vectorise.expand_globs",
                return_value=configs.files,
            ),
            patch("vectorcode.subcommands.vectorise.chunked_add") as mock_chunked_add,
        ):
            MockClientManager.return_value._create_client.return_value = mock_client
            await vectorise(configs)
            # Assert that chunked_add is only called for test_file.py, not excluded_file.py
            call_args = [call[0][0] for call in mock_chunked_add.call_args_list]
            assert str(os.path.join(tmpdir, "excluded_file.py")) not in call_args
            assert os.path.join(tmpdir, "test_file.py") in call_args
            assert mock_chunked_add.call_count == 2


@pytest.mark.asyncio
async def test_vectorise_exclude_file_recursive():
    # Create a temporary .vectorcode directory and vectorcode.exclude file
    with tempfile.TemporaryDirectory() as tmpdir:
        exclude_dir = os.path.join(tmpdir, ".vectorcode")
        nested_dir = os.path.join(tmpdir, "nested")

        os.makedirs(exclude_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)

        exclude_spec = os.path.join(exclude_dir, "vectorcode.exclude")
        with open(exclude_spec, mode="w") as fin:
            fin.writelines(["excluded_file.py"])
        with open(os.path.join(nested_dir, ".gitignore"), "w") as fin:
            fin.writelines(["excluded_file.py"])
        with open(os.path.join(nested_dir, "excluded_file.py"), "w") as fin:
            # recursive case. This file should be skipped.
            fin.writelines(['print("hello world")'])

        configs = Config(
            db_url="http://test_host:1234",
            db_path="test_db",
            embedding_function="SentenceTransformerEmbeddingFunction",
            embedding_params={},
            project_root=str(tmpdir),
            files=[
                os.path.join(tmpdir, "test_file.py"),
                os.path.join(tmpdir, "excluded_file.py"),
            ],
            recursive=True,
            force=False,
            pipe=False,
        )
        mock_client = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.get.return_value = {"ids": []}

        with (
            patch(
                "vectorcode.subcommands.vectorise.ClientManager"
            ) as MockClientManager,
            patch(
                "vectorcode.subcommands.vectorise.get_collection",
                return_value=mock_collection,
            ),
            patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
            patch(
                "vectorcode.subcommands.vectorise.expand_globs",
                return_value=configs.files,
            ),
            patch("vectorcode.subcommands.vectorise.chunked_add") as mock_chunked_add,
        ):
            MockClientManager.return_value._create_client.return_value = mock_client
            await vectorise(configs)
            # Assert that chunked_add is only called for test_file.py, not excluded_file.py
            call_args = [call[0][0] for call in mock_chunked_add.call_args_list]
            assert str(os.path.join(tmpdir, "excluded_file.py")) not in call_args
            assert os.path.join(tmpdir, "test_file.py") in call_args
            assert mock_chunked_add.call_count == 1


@pytest.mark.asyncio
async def test_vectorise_uses_global_exclude_when_local_missing():
    mock_client = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {"ids": []}

    with tempfile.TemporaryDirectory() as temp_home:
        os.environ["HOME"] = temp_home
        global_config_dir = os.path.join(temp_home, ".config", "vectorcode")
        os.makedirs(global_config_dir, exist_ok=True)
        with open(
            os.path.join(global_config_dir, "vectorcode.exclude"), mode="w"
        ) as fin:
            fin.writelines(["exclude.py"])

        project_root = os.path.join(temp_home, "project")
        os.makedirs(project_root, exist_ok=True)
        files = list(
            os.path.join(project_root, i) for i in ("include.py", "exclude.py")
        )
        for f_name in files:
            full_path = os.path.join(project_root, f_name)
            with open(full_path, mode="w") as fin:
                pass
        with (
            patch(
                "vectorcode.subcommands.vectorise.ClientManager"
            ) as MockClientManager,
            patch(
                "vectorcode.subcommands.vectorise.get_collection",
                return_value=mock_collection,
            ),
            patch("vectorcode.subcommands.vectorise.verify_ef", return_value=True),
            patch("vectorcode.subcommands.vectorise.chunked_add") as mock_chunked_add,
            patch(
                "vectorcode.subcommands.vectorise.GLOBAL_EXCLUDE_SPEC",
                os.path.join(temp_home, ".config", "vectorcode", "vectorcode.exclude"),
            ),
        ):
            MockClientManager.return_value._create_client.return_value = mock_client
            await vectorise(
                Config(
                    project_root=project_root,
                    files=list(os.path.join(project_root, i) for i in files),
                    action=CliAction.vectorise,
                )
            )
            mock_chunked_add.assert_called_once()
