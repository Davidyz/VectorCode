from unittest.mock import AsyncMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.subcommands.drop import drop


@pytest.mark.asyncio
async def test_drop_success():
    mock_db = AsyncMock()
    with patch(
        "vectorcode.subcommands.drop.get_database_connector", return_value=mock_db
    ):
        await drop(config=Config(project_root="DummyDir"))
        mock_db.drop.assert_called_once()


@pytest.mark.asyncio
async def test_drop_collection_not_found():
    mock_db = AsyncMock()
    mock_db.drop = AsyncMock(side_effect=CollectionNotFoundError)
    with patch(
        "vectorcode.subcommands.drop.get_database_connector", return_value=mock_db
    ):
        assert await drop(config=Config(project_root="DummyDir")) != 0
