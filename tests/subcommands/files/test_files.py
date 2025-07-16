from unittest.mock import AsyncMock, patch

import pytest

from vectorcode.cli_utils import CliAction, Config, FilesAction
from vectorcode.subcommands.files import files


@pytest.mark.asyncio
async def test_files():
    with patch(
        "vectorcode.subcommands.files.ls.ls", return_value=AsyncMock()
    ) as mock_ls:
        config = Config(action=CliAction.files, files_action=FilesAction.ls)
        await files(config)
        mock_ls.assert_called_with(config)
    with patch(
        "vectorcode.subcommands.files.rm.rm", return_value=AsyncMock()
    ) as mock_rm:
        config = Config(action=CliAction.files, files_action=FilesAction.rm)
        await files(config)
        mock_rm.assert_called_with(config)


@pytest.mark.asyncio
async def test_files_invalid_actions():
    with patch("vectorcode.subcommands.files.logger") as mock_logger:
        config = Config(action=CliAction.files, files_action="foobar")
        assert await files(config) != 0
        mock_logger.error.assert_called_once()
