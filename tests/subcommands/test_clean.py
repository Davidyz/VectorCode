from unittest.mock import AsyncMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.subcommands.clean import clean


@pytest.mark.asyncio
async def test_clean(capsys):
    mock_db = AsyncMock()
    mock_db.cleanup.return_value = ["/test/path1", "/test/path2"]

    with patch(
        "vectorcode.subcommands.clean.get_database_connector", return_value=mock_db
    ):
        result = await clean(Config(pipe=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "Deleted collection: /test/path1" in captured.out
    assert "Deleted collection: /test/path2" in captured.out


@pytest.mark.asyncio
async def test_clean_pipe_mode(capsys):
    mock_db = AsyncMock()
    mock_db.cleanup.return_value = ["/test/path1", "/test/path2"]

    with patch(
        "vectorcode.subcommands.clean.get_database_connector", return_value=mock_db
    ):
        result = await clean(Config(pipe=True))

    assert result == 0
    captured = capsys.readouterr()
    assert captured.out == ""
