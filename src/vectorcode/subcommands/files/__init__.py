import logging

from vectorcode.cli_utils import Config, FilesAction

logger = logging.getLogger(name=__name__)


async def files(configs: Config) -> int:
    match configs.files_action:
        case FilesAction.ls:
            from vectorcode.subcommands.files import ls

            return await ls.ls(configs)
        case FilesAction.rm:
            from vectorcode.subcommands.files import rm

            return await rm.rm(configs)
        case _:
            logger.error(
                f"Unsupported subcommand for `vectorcode files`: {configs.action}"
            )
            return 1
