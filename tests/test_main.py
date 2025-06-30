from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vectorcode import __version__
from vectorcode.cli_utils import CliAction, Config
from vectorcode.main import async_main


@pytest.mark.asyncio
async def test_async_main_no_stderr(monkeypatch):
    mock_cli_args = MagicMock(
        no_stderr=True, project_root=".", action=CliAction.version
    )
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )

    with patch("os.devnull", "/dev/null"):
        with patch("sys.stderr") as mock_stderr:
            await async_main()
            mock_stderr.assert_not_called()


@pytest.mark.asyncio
async def test_async_main_default_project_root(monkeypatch):
    mock_cli_args = MagicMock(
        no_stderr=False, project_root=None, action=CliAction.version
    )
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    monkeypatch.setattr("os.getcwd", MagicMock(return_value="/test/cwd"))
    monkeypatch.setattr(
        "vectorcode.main.find_project_root", MagicMock(return_value="/test/cwd")
    )

    await async_main()

    assert mock_cli_args.project_root == "/test/cwd"


@pytest.mark.asyncio
async def test_async_main_ioerror(monkeypatch):
    mock_cli_args = MagicMock(
        no_stderr=False, project_root=".", action=CliAction.version
    )
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(side_effect=IOError("Test Error")),
    )

    with patch("sys.stderr.write") as mock_stderr:
        return_code = await async_main()
        assert return_code == 1
        mock_stderr.assert_called()


@pytest.mark.asyncio
async def test_async_main_cli_action_check(monkeypatch):
    mock_cli_args = MagicMock(no_stderr=False, project_root=".", action=CliAction.check)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_check = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.check", mock_check)
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(return_value=MagicMock(merge_from=AsyncMock())),
    )

    return_code = await async_main()
    assert return_code == 0
    mock_check.assert_called_once()


@pytest.mark.asyncio
async def test_async_main_cli_action_init(monkeypatch):
    mock_cli_args = MagicMock(no_stderr=False, project_root=".", action=CliAction.init)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_init = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.init", mock_init)
    monkeypatch.setattr("vectorcode.main.get_project_config", AsyncMock())

    return_code = await async_main()
    assert return_code == 0
    mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_async_main_cli_action_chunks(monkeypatch):
    mock_cli_args = MagicMock(
        no_stderr=False, project_root=".", action=CliAction.chunks
    )
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_chunks = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.chunks", mock_chunks)
    monkeypatch.setattr("vectorcode.main.get_project_config", AsyncMock())
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))

    return_code = await async_main()
    assert return_code == 0
    mock_chunks.assert_called_once()


@pytest.mark.asyncio
async def test_async_main_cli_action_version(monkeypatch, capsys):
    mock_cli_args = MagicMock(
        no_stderr=False, project_root=".", action=CliAction.version
    )
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )

    return_code = await async_main()
    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.out.strip() == __version__


@pytest.mark.asyncio
async def test_async_main_cli_action_prompts(monkeypatch):
    mock_cli_args = MagicMock(project_root=".", action=CliAction.prompts)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_prompts = MagicMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.prompts", mock_prompts)
    monkeypatch.setattr("vectorcode.main.get_project_config", AsyncMock())

    return_code = await async_main()
    assert return_code == 0
    mock_prompts.assert_called_once()


@pytest.mark.asyncio
async def test_async_main_cli_action_query(monkeypatch):
    mock_cli_args = MagicMock(no_stderr=False, project_root=".", action=CliAction.query)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_final_configs = MagicMock(
        host="test_host", port=1234, action=CliAction.query, pipe=False
    )
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(
            return_value=AsyncMock(
                merge_from=AsyncMock(return_value=mock_final_configs)
            )
        ),
    )
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))
    mock_query = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.query", mock_query)

    return_code = await async_main()
    assert return_code == 0
    mock_query.assert_called_once_with(mock_final_configs)


@pytest.mark.asyncio
async def test_async_main_cli_action_vectorise(monkeypatch):
    mock_cli_args = MagicMock(
        no_stderr=False,
        project_root=".",
        action=CliAction.vectorise,
        include_hidden=True,
    )
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_final_configs = MagicMock(
        host="test_host", port=1234, action=CliAction.vectorise, include_hidden=True
    )
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(
            return_value=AsyncMock(
                merge_from=AsyncMock(return_value=mock_final_configs)
            )
        ),
    )
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))
    mock_vectorise = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.vectorise", mock_vectorise)

    return_code = await async_main()
    assert return_code == 0
    mock_vectorise.assert_called_once_with(mock_final_configs)


@pytest.mark.asyncio
async def test_async_main_cli_action_drop(monkeypatch):
    mock_cli_args = MagicMock(no_stderr=False, project_root=".", action=CliAction.drop)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_final_configs = MagicMock(host="test_host", port=1234, action=CliAction.drop)
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(
            return_value=AsyncMock(
                merge_from=AsyncMock(return_value=mock_final_configs)
            )
        ),
    )
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))
    mock_drop = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.drop", mock_drop)

    return_code = await async_main()
    assert return_code == 0
    mock_drop.assert_called_once_with(mock_final_configs)


@pytest.mark.asyncio
async def test_async_main_cli_action_ls(monkeypatch):
    mock_cli_args = MagicMock(no_stderr=False, project_root=".", action=CliAction.ls)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_final_configs = MagicMock(host="test_host", port=1234, action=CliAction.ls)
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(
            return_value=AsyncMock(
                merge_from=AsyncMock(return_value=mock_final_configs)
            )
        ),
    )
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))
    mock_ls = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.ls", mock_ls)

    return_code = await async_main()
    assert return_code == 0
    mock_ls.assert_called_once_with(mock_final_configs)


@pytest.mark.asyncio
async def test_async_main_cli_action_files(monkeypatch):
    cli_args = Config(action=CliAction.files)
    mock_files = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.files", mock_files)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=cli_args)
    )
    assert await async_main() == 0
    mock_files.assert_called_once()


@pytest.mark.asyncio
async def test_async_main_cli_action_update(monkeypatch):
    mock_cli_args = MagicMock(
        no_stderr=False, project_root=".", action=CliAction.update
    )
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_final_configs = MagicMock(host="test_host", port=1234, action=CliAction.update)
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(
            return_value=AsyncMock(
                merge_from=AsyncMock(return_value=mock_final_configs)
            )
        ),
    )
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))
    mock_update = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.update", mock_update)

    return_code = await async_main()
    assert return_code == 0
    mock_update.assert_called_once_with(mock_final_configs)


@pytest.mark.asyncio
async def test_async_main_cli_action_clean(monkeypatch):
    mock_cli_args = MagicMock(no_stderr=False, project_root=".", action=CliAction.clean)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_final_configs = MagicMock(host="test_host", port=1234, action=CliAction.clean)
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(
            return_value=AsyncMock(
                merge_from=AsyncMock(return_value=mock_final_configs)
            )
        ),
    )
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))
    mock_clean = AsyncMock(return_value=0)
    monkeypatch.setattr("vectorcode.subcommands.clean", mock_clean)

    return_code = await async_main()
    assert return_code == 0
    mock_clean.assert_called_once_with(mock_final_configs)


@pytest.mark.asyncio
async def test_async_main_exception_handling(monkeypatch):
    mock_cli_args = MagicMock(no_stderr=False, project_root=".", action=CliAction.query)
    monkeypatch.setattr(
        "vectorcode.main.parse_cli_args", AsyncMock(return_value=mock_cli_args)
    )
    mock_final_configs = MagicMock(host="test_host", port=1234, action=CliAction.query)
    monkeypatch.setattr(
        "vectorcode.main.get_project_config",
        AsyncMock(
            return_value=AsyncMock(
                merge_from=AsyncMock(return_value=mock_final_configs)
            )
        ),
    )
    monkeypatch.setattr("vectorcode.common.try_server", AsyncMock(return_value=True))
    mock_query = AsyncMock(side_effect=Exception("Test Exception"))
    monkeypatch.setattr("vectorcode.subcommands.query", mock_query)

    with patch("vectorcode.main.logger") as mock_logger:
        assert await async_main() == 1
        mock_logger.error.assert_called_once()
