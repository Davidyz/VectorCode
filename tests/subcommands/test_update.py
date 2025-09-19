import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.database.types import FileInCollection
from vectorcode.subcommands.update import update


@pytest.mark.asyncio
async def test_update_success(tmp_path):
    """Test successful update with some modified files."""
    config = Config(project_root=str(tmp_path), pipe=False)

    # Mock files in the database
    file1_path = tmp_path / "file1.py"
    file1_path.write_text("content1")
    file2_path = tmp_path / "file2.py"
    file2_path.write_text("new content2")  # modified
    file3_path = tmp_path / "file3.py"
    file3_path.write_text("content3")

    collection_files = [
        FileInCollection(path=str(file1_path), sha256="hash1_old"),
        FileInCollection(path=str(file2_path), sha256="hash2_old"),
        FileInCollection(path=str(file3_path), sha256="hash3_old"),
    ]

    with (
        patch("vectorcode.subcommands.update.get_database_connector") as mock_get_db,
        patch(
            "vectorcode.subcommands.update.vectorise_worker", new_callable=AsyncMock
        ) as mock_vectorise_worker,
        patch("vectorcode.subcommands.update.show_stats") as mock_show_stats,
        patch("vectorcode.subcommands.update.hash_file") as mock_hash_file,
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.return_value.files = collection_files
        mock_get_db.return_value = mock_db

        # file1.py is unchanged, file2.py is changed, file3.py is unchanged
        mock_hash_file.side_effect = ["hash1_old", "hash2_new", "hash3_old"]

        result = await update(config)

        assert result == 0
        mock_db.list_collection_content.assert_called_once()

        # vectorise_worker should only be called for the modified file (file2.py)
        assert mock_vectorise_worker.call_count == 1
        # Check that it was called with file2.py
        called_with_file = mock_vectorise_worker.call_args_list[0][0][1]
        assert called_with_file == str(file2_path)

        mock_db.check_orphanes.assert_called_once()
        mock_show_stats.assert_called_once()


@pytest.mark.asyncio
async def test_update_force(tmp_path):
    """Test update with force=True, all files should be re-vectorised."""
    config = Config(project_root=str(tmp_path), pipe=False, force=True)

    file1_path = tmp_path / "file1.py"
    file1_path.write_text("content1")
    file2_path = tmp_path / "file2.py"
    file2_path.write_text("content2")

    collection_files = [
        FileInCollection(path=str(file1_path), sha256="hash1"),
        FileInCollection(path=str(file2_path), sha256="hash2"),
    ]

    with (
        patch("vectorcode.subcommands.update.get_database_connector") as mock_get_db,
        patch(
            "vectorcode.subcommands.update.vectorise_worker", new_callable=AsyncMock
        ) as mock_vectorise_worker,
        patch("vectorcode.subcommands.update.show_stats") as mock_show_stats,
        patch("vectorcode.subcommands.update.hash_file") as mock_hash_file,
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.return_value.files = collection_files
        mock_get_db.return_value = mock_db

        result = await update(config)

        assert result == 0
        mock_db.list_collection_content.assert_called_once()

        # vectorise_worker should be called for all files
        assert mock_vectorise_worker.call_count == 2
        mock_hash_file.assert_not_called()  # hash_file should not be called with force=True

        mock_db.check_orphanes.assert_called_once()
        mock_show_stats.assert_called_once()


@pytest.mark.asyncio
async def test_update_cancelled(tmp_path):
    """Test update being cancelled."""
    config = Config(project_root=str(tmp_path), pipe=False)

    file1_path = tmp_path / "file1.py"
    file1_path.write_text("content1")

    collection_files = [
        FileInCollection(path=str(file1_path), sha256="hash1_old"),
    ]

    with (
        patch("vectorcode.subcommands.update.get_database_connector") as mock_get_db,
        patch(
            "vectorcode.subcommands.update.vectorise_worker", new_callable=AsyncMock
        ) as mock_vectorise_worker,
        patch("vectorcode.subcommands.update.hash_file", return_value="hash1_new"),
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.return_value.files = collection_files
        mock_get_db.return_value = mock_db

        mock_vectorise_worker.side_effect = asyncio.CancelledError

        result = await update(config)

        assert result == 1
        mock_db.check_orphanes.assert_not_called()


@pytest.mark.asyncio
async def test_update_empty_collection(tmp_path):
    """Test update with an empty collection."""
    config = Config(project_root=str(tmp_path), pipe=False)

    with (
        patch("vectorcode.subcommands.update.get_database_connector") as mock_get_db,
        patch(
            "vectorcode.subcommands.update.vectorise_worker", new_callable=AsyncMock
        ) as mock_vectorise_worker,
        patch("vectorcode.subcommands.update.show_stats") as mock_show_stats,
    ):
        mock_db = AsyncMock()
        mock_db.list_collection_content.return_value.files = []
        mock_get_db.return_value = mock_db

        result = await update(config)

        assert result == 0
        mock_vectorise_worker.assert_not_called()
        mock_db.check_orphanes.assert_called_once()
        mock_show_stats.assert_called_once()
