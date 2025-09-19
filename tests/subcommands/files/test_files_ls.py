import json
from unittest.mock import AsyncMock, patch

import pytest

from vectorcode.cli_utils import CliAction, Config, FilesAction
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.database.types import FileInCollection
from vectorcode.subcommands.files.ls import ls


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.list_collection_content.return_value.files = [
        FileInCollection(path="file1.py", sha256="hash1"),
        FileInCollection(path="file2.py", sha256="hash2"),
        FileInCollection(path="file3.py", sha256="hash3"),
    ]
    return db


@pytest.mark.asyncio
async def test_ls(mock_db, capsys):
    with patch(
        "vectorcode.subcommands.files.ls.get_database_connector", return_value=mock_db
    ):
        await ls(Config(action=CliAction.files, files_action=FilesAction.ls))
        out = capsys.readouterr().out
        assert "file1.py" in out
        assert "file2.py" in out
        assert "file3.py" in out


@pytest.mark.asyncio
async def test_ls_piped(mock_db, capsys):
    with patch(
        "vectorcode.subcommands.files.ls.get_database_connector", return_value=mock_db
    ):
        await ls(Config(action=CliAction.files, files_action=FilesAction.ls, pipe=True))
        out = capsys.readouterr().out
        assert json.dumps(["file1.py", "file2.py", "file3.py"]).strip() == out.strip()


@pytest.mark.asyncio
async def test_ls_no_collection(mock_db):
    mock_db.list_collection_content.side_effect = CollectionNotFoundError
    with patch(
        "vectorcode.subcommands.files.ls.get_database_connector", return_value=mock_db
    ):
        assert (
            await ls(
                Config(action=CliAction.files, files_action=FilesAction.ls, pipe=True)
            )
            != 0
        )


@pytest.mark.asyncio
async def test_ls_empty_collection(mock_db, capsys):
    mock_db.list_collection_content.return_value.files = []
    with patch(
        "vectorcode.subcommands.files.ls.get_database_connector", return_value=mock_db
    ):
        assert (
            await ls(
                Config(pipe=True, action=CliAction.files, files_action=FilesAction.ls)
            )
            == 0
        )
        assert capsys.readouterr().out.strip() == "[]"
