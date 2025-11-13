import hashlib
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.database.types import VectoriseStats
from vectorcode.database.utils import get_uuid, hash_file, hash_str
from vectorcode.subcommands.vectorise import (
    find_exclude_specs,
    load_files_from_include,
    vectorise,
)


def test_hash_str():
    test_string = "test_string"
    expected_hash = hashlib.sha256(test_string.encode()).hexdigest()
    assert hash_str(test_string) == expected_hash


def test_hash_file(tmp_path):
    content = b"This is a test file for hashing."
    expected_hash = hashlib.sha256(content).hexdigest()
    file_path = tmp_path / "test_file.txt"
    file_path.write_bytes(content)

    actual_hash = hash_file(str(file_path))
    assert actual_hash == expected_hash


def test_get_uuid():
    uuid_str = get_uuid()
    assert isinstance(uuid_str, str)
    assert len(uuid_str) == 32  # UUID4 hex string length


@patch("tabulate.tabulate")
def test_show_stats_pipe_false(mock_tabulate, capsys):
    from vectorcode.subcommands.vectorise import show_stats

    configs = Config(pipe=False)
    stats = VectoriseStats(**{"add": 1, "update": 2, "removed": 3})
    show_stats(configs, stats)
    mock_tabulate.assert_called_once()


def test_show_stats_pipe_true(capsys):
    from vectorcode.subcommands.vectorise import show_stats

    configs = Config(pipe=True)
    stats = VectoriseStats(**{"add": 1, "update": 2, "removed": 3})
    show_stats(configs, stats)
    captured = capsys.readouterr()
    assert captured.out.strip() == (stats.to_json())


@pytest.mark.asyncio
async def test_vectorise_success(tmp_path):
    config = Config(project_root=str(tmp_path), files=["file1.py", "file2.py"])
    with (
        patch("vectorcode.subcommands.vectorise.get_database_connector") as mock_get_db,
        patch(
            "vectorcode.subcommands.vectorise.expand_globs", return_value=config.files
        ),
        patch("vectorcode.subcommands.vectorise.FilterManager") as mock_filter_manager,
        patch(
            "vectorcode.subcommands.vectorise.vectorise_worker", new_callable=AsyncMock
        ) as mock_worker,
        patch("vectorcode.subcommands.vectorise.show_stats") as mock_show_stats,
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.return_value.files = []
        mock_get_db.return_value = mock_db
        mock_filter_manager.return_value.return_value = config.files

        result = await vectorise(config)

        assert result == 0
        assert mock_worker.call_count == 2
        mock_show_stats.assert_called_once()


@pytest.mark.asyncio
async def test_vectorise_with_excludes(tmp_path):
    config = Config(project_root=str(tmp_path), files=["file1.py", "file2.py"])
    with (
        patch("vectorcode.subcommands.vectorise.get_database_connector") as mock_get_db,
        patch(
            "vectorcode.subcommands.vectorise.expand_globs", return_value=config.files
        ),
        patch("vectorcode.subcommands.vectorise.FilterManager") as mock_filter_manager,
        patch(
            "vectorcode.subcommands.vectorise.vectorise_worker", new_callable=AsyncMock
        ) as mock_worker,
        patch(
            "vectorcode.subcommands.vectorise.find_exclude_specs",
            return_value=[".gitignore"],
        ),
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.return_value.files = []
        mock_get_db.return_value = mock_db
        mock_filter_manager.return_value.return_value = ["file1.py"]

        await vectorise(config)

        assert mock_worker.call_count == 1


@pytest.mark.asyncio
async def test_vectorise_collection_not_found(tmp_path):
    full_path = os.path.join(tmp_path, "file1.py")
    config = Config(project_root=str(tmp_path), files=[full_path])
    Path(full_path).touch()
    with (
        patch("vectorcode.subcommands.vectorise.get_database_connector") as mock_get_db,
        patch(
            "vectorcode.subcommands.vectorise.expand_globs", return_value=config.files
        ),
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.side_effect = CollectionNotFoundError
        mock_get_db.return_value = mock_db

        # This should not raise an exception
        await vectorise(config)


def test_find_exclude_specs(tmp_path):
    config = Config(project_root=str(tmp_path), recursive=True)
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.touch()
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_gitignore_path = nested_dir / ".gitignore"
    nested_gitignore_path.touch()

    specs = find_exclude_specs(config)
    assert str(gitignore_path) in specs
    assert str(nested_gitignore_path) in specs


@patch("os.path.isfile", return_value=True)
@patch("pathspec.PathSpec.check_tree_files")
def test_load_files_from_local_include(mock_check_tree_files, mock_isfile, tmp_path):
    project_root = str(tmp_path)
    local_include_dir = tmp_path / ".vectorcode"
    local_include_dir.mkdir()
    local_include_file = local_include_dir / "vectorcode.include"
    local_include_content = "local_file1.py\nlocal_file2.py"
    local_include_file.write_text(local_include_content)

    mock_isfile.side_effect = lambda p: str(p) == str(local_include_file)

    mock_check_tree_files.return_value = [
        MagicMock(file="local_file1.py", include=True),
        MagicMock(file="local_file2.py", include=True),
        MagicMock(file="ignored_file.py", include=False),
    ]

    m_open = MagicMock()
    m_open.return_value.__enter__.return_value.readlines.return_value = (
        local_include_content.splitlines()
    )
    with patch("builtins.open", m_open):
        files = load_files_from_include(project_root)

    assert "local_file1.py" in files
    assert "local_file2.py" in files
    assert "ignored_file.py" not in files


def test_find_exclude_specs_non_recursive(tmp_path):
    config = Config(project_root=str(tmp_path), recursive=False)
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.touch()
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_gitignore_path = nested_dir / ".gitignore"
    nested_gitignore_path.touch()

    specs = find_exclude_specs(config)
    assert str(gitignore_path) in specs
    assert str(nested_gitignore_path) not in specs


@patch("os.path.isfile")
def test_find_exclude_specs_global(mock_isfile, tmp_path):
    from vectorcode.subcommands.vectorise import GLOBAL_EXCLUDE_SPEC

    config = Config(project_root=str(tmp_path), recursive=False)

    def isfile_side_effect(path):
        if path == GLOBAL_EXCLUDE_SPEC:
            return True
        return os.path.join(str(tmp_path), ".gitignore") == path

    mock_isfile.side_effect = isfile_side_effect

    specs = find_exclude_specs(config)
    assert GLOBAL_EXCLUDE_SPEC in specs


def test_find_exclude_specs_non_recursive_no_gitignore(tmp_path):
    config = Config(project_root=str(tmp_path), recursive=False)
    specs = find_exclude_specs(config)
    assert specs == []


def test_find_exclude_specs_local_exclude(tmp_path):
    config = Config(project_root=str(tmp_path), recursive=False)
    exclude_dir = tmp_path / ".vectorcode"
    exclude_dir.mkdir()
    exclude_file = exclude_dir / "vectorcode.exclude"
    exclude_file.touch()

    specs = find_exclude_specs(config)
    assert str(exclude_file) in specs


@patch("os.path.isfile")
@patch("pathspec.PathSpec.check_tree_files")
def test_load_files_from_global_include(mock_check_tree_files, mock_isfile, tmp_path):
    from vectorcode.subcommands.vectorise import GLOBAL_INCLUDE_SPEC

    project_root = str(tmp_path)
    global_include_content = "global_file1.py\nglobal_file2.py"

    def isfile_side_effect(p):
        return str(p) == GLOBAL_INCLUDE_SPEC

    mock_isfile.side_effect = isfile_side_effect

    mock_check_tree_files.return_value = [
        MagicMock(file="global_file1.py", include=True),
        MagicMock(file="global_file2.py", include=True),
        MagicMock(file="ignored_file.py", include=False),
    ]

    m_open = MagicMock()
    m_open.return_value.__enter__.return_value.readlines.return_value = (
        global_include_content.splitlines()
    )
    with patch("builtins.open", m_open):
        files = load_files_from_include(project_root)

    assert "global_file1.py" in files
    assert "global_file2.py" in files
    assert "ignored_file.py" not in files


@patch("os.path.isfile", return_value=False)
def test_load_files_from_include_no_spec(mock_isfile, tmp_path):
    project_root = str(tmp_path)
    files = load_files_from_include(project_root)
    assert files == []
