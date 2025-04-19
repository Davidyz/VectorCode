import asyncio
import logging
import os
import sys
import traceback

from vectorcode import __version__
from vectorcode.cli_utils import (
    CliAction,
    config_logging,
    find_project_root,
    get_project_config,
    parse_cli_args,
)

logger = logging.getLogger(name=__name__)


async def async_main():
    cli_args = await parse_cli_args()
    if cli_args.no_stderr:
        sys.stderr = open(os.devnull, "w")
    logger.info("Collected CLI arguments: %s", cli_args)

    if cli_args.project_root is None:
        cwd = os.getcwd()
        cli_args.project_root = (
            find_project_root(cwd, ".vectorcode")
            or find_project_root(cwd, ".git")
            or cwd
        )

    logger.info(f"Project root is set to {cli_args.project_root}")

    try:
        final_configs = await (
            await get_project_config(cli_args.project_root)
        ).merge_from(cli_args)
    except IOError as e:
        traceback.print_exception(e, file=sys.stderr)
        return 1

    logger.info("Final configuration has been built: %s", final_configs)

    match cli_args.action:
        case CliAction.check:
            from vectorcode.subcommands import check

            return await check(cli_args)
        case CliAction.init:
            from vectorcode.subcommands import init

            return await init(cli_args)
        case CliAction.version:
            print(__version__)
            return 0
        case CliAction.prompts:
            from vectorcode.subcommands import prompts

            return prompts(cli_args)
        case CliAction.chunks:
            from vectorcode.subcommands import chunks

            return_val = await chunks(final_configs)

    from vectorcode.common import start_server, try_server

    server_process = None
    if not (
        await try_server(final_configs.host, final_configs.port, True)
        or (await try_server(final_configs.host, final_configs.port, False))
    ):
        server_process = await start_server(final_configs)

    if final_configs.pipe:
        # NOTE: NNCF (intel GPU acceleration for sentence transformer) keeps showing logs.
        # This disables logs below ERROR so that it doesn't hurt the `pipe` output.
        logging.disable(logging.ERROR)

    return_val = 0
    try:
        match final_configs.action:
            case CliAction.query:
                from vectorcode.subcommands import query

                return_val = await query(final_configs)
            case CliAction.vectorise:
                from vectorcode.subcommands import vectorise

                return_val = await vectorise(final_configs)
            case CliAction.drop:
                from vectorcode.subcommands import drop

                return_val = await drop(final_configs)
            case CliAction.ls:
                from vectorcode.subcommands import ls

                return_val = await ls(final_configs)
            case CliAction.update:
                from vectorcode.subcommands import update

                return_val = await update(final_configs)
            case CliAction.clean:
                from vectorcode.subcommands import clean

                return_val = await clean(final_configs)
    except Exception as e:
        return_val = 1
        traceback.print_exception(e, file=sys.stderr)
        logger.error(traceback.format_exc())
    finally:
        if server_process is not None:
            logger.info("Shutting down the bundled Chromadb instance.")
            server_process.terminate()
            await server_process.wait()
        return return_val


def main():  # pragma: nocover
    config_logging("vectorcode")
    return asyncio.run(async_main())


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
