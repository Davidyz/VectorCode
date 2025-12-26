from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.subcommands.files.rm import rm


@pytest.fixture
def mock_db():
    db = AsyncMock()

    def mock_delete():
        count = 0
        for f in db._configs.rm_paths:
            if Path(f).name in {"file1.py", "file2.py", "file3.py"}:
                count += 1
        return count

    db.delete = AsyncMock(side_effect=mock_delete)
    return db


@pytest.mark.asyncio
async def test_rm(mock_db, capsys):
    configs = Config(rm_paths=["file1.py", "file2.py"])
    mock_db._configs = configs
    with patch(
        "vectorcode.subcommands.files.rm.get_database_connector", return_value=mock_db
    ):
        assert await rm(configs) == 0
        assert capsys.readouterr().out.strip() == "Removed 2 file(s)."


@pytest.mark.asyncio
async def test_rm_clean_after_rm(mock_db, capsys):
    configs = Config(rm_paths=["file1.py", "file2.py"])
    mock_db._configs = configs
    mock_db.count = AsyncMock(return_value=0)
    with patch(
        "vectorcode.subcommands.files.rm.get_database_connector", return_value=mock_db
    ):
        assert await rm(configs) == 0
        mock_db.drop.assert_called_once()


@pytest.mark.asyncio
async def test_rm_no_collection(mock_db, capsys):
    with patch(
        "vectorcode.subcommands.files.rm.get_database_connector", return_value=mock_db
    ):
        mock_db.delete.side_effect = CollectionNotFoundError
        assert (
            await rm(
                Config(
                    rm_paths=["file1.py"],
                )
            )
            != 0
        )
