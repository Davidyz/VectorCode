import argparse
import asyncio
import logging
import os
import sys
import time
import uuid

try:  # pragma: nocover
    from lsprotocol import types
    from pygls.server import LanguageServer
except ModuleNotFoundError:  # pragma: nocover
    print(
        "Please install the `vectorcode[lsp]` dependency group to use the LSP feature.",
        file=sys.stderr,
    )
    sys.exit(1)
from vectorcode import __version__
from vectorcode.cli_utils import (
    CliAction,
    Config,
    cleanup_path,
    config_logging,
    find_project_root,
    get_project_config,
    parse_cli_args,
)
from vectorcode.common import get_client, get_collection, try_server
from vectorcode.subcommands.ls import get_collection_list
from vectorcode.subcommands.query import build_query_results

cached_project_configs: dict[str, Config] = {}
DEFAULT_PROJECT_ROOT: str | None = None
logger = logging.getLogger(__name__)


async def make_caches(project_root: str):
    assert os.path.isabs(project_root)
    if cached_project_configs.get(project_root) is None:
        cached_project_configs[project_root] = await get_project_config(project_root)
    config = cached_project_configs[project_root]
    config.project_root = project_root
    db_url = config.db_url
    if not await try_server(db_url):  # pragma: nocover
        raise ConnectionError(
            "Failed to find an existing ChromaDB server, which is a hard requirement for LSP mode!"
        )


def get_arg_parser():
    parser = argparse.ArgumentParser(
        "vectorcode-server", description="VectorCode LSP daemon."
    )
    parser.add_argument("--version", action="store_true", default=False)
    parser.add_argument(
        "--project_root",
        help="Default project root for VectorCode queries.",
        type=str,
        default="",
    )
    return parser


server: LanguageServer = LanguageServer(name="vectorcode-server", version=__version__)


@server.command("vectorcode")
async def execute_command(ls: LanguageServer, args: list[str]):
    global DEFAULT_PROJECT_ROOT
    start_time = time.time()
    logger.info("Received command arguments: %s", args)
    parsed_args = await parse_cli_args(args)
    logger.info("Parsed command arguments: %s", parsed_args)
    if parsed_args.action not in {CliAction.query, CliAction.ls}:
        print(
            f"Unsupported vectorcode subcommand: {str(parsed_args.action)}",
            file=sys.stderr,
        )
        return
    if parsed_args.project_root is None:
        if DEFAULT_PROJECT_ROOT is not None:
            parsed_args.project_root = DEFAULT_PROJECT_ROOT
            logger.warning("Using DEFAULT_PROJECT_ROOT: %s", DEFAULT_PROJECT_ROOT)
    elif DEFAULT_PROJECT_ROOT is None:
        logger.warning("Updating DEFAULT_PROJECT_ROOT to %s", parsed_args.project_root)
        DEFAULT_PROJECT_ROOT = str(parsed_args.project_root)

    if parsed_args.project_root is not None:
        parsed_args.project_root = os.path.abspath(str(parsed_args.project_root))
        await make_caches(parsed_args.project_root)
        final_configs = await cached_project_configs[
            parsed_args.project_root
        ].merge_from(parsed_args)
        final_configs.pipe = True
        client = await get_client(final_configs)
        collection = await get_collection(
            client=client,
            configs=final_configs,
            make_if_missing=final_configs.action in {CliAction.vectorise},
        )
    else:
        final_configs = parsed_args
        client = await get_client(parsed_args)
        collection = None
    logger.info("Merged final configs: %s", final_configs)
    progress_token = str(uuid.uuid4())

    await ls.progress.create_async(progress_token)
    match final_configs.action:
        case CliAction.query:
            ls.progress.begin(
                progress_token,
                types.WorkDoneProgressBegin(
                    "VectorCode",
                    message=f"Querying {cleanup_path(str(final_configs.project_root))}",
                ),
            )
            final_results = []
            try:
                if collection is None:
                    print("Please specify a project to search in.", file=sys.stderr)
                else:
                    final_results.extend(
                        await build_query_results(collection, final_configs)
                    )
            finally:
                log_message = f"Retrieved {len(final_results)} result{'s' if len(final_results) > 1 else ''} in {round(time.time() - start_time, 2)}s."
                ls.progress.end(
                    progress_token,
                    types.WorkDoneProgressEnd(message=log_message),
                )
                logger.info(log_message)
            return final_results
        case CliAction.ls:
            ls.progress.begin(
                progress_token,
                types.WorkDoneProgressBegin(
                    "VectorCode",
                    message="Looking for other projects indexed by VectorCode",
                ),
            )
            projects: list[dict] = []
            try:
                projects.extend(await get_collection_list(client))
            finally:
                ls.progress.end(
                    progress_token,
                    types.WorkDoneProgressEnd(message="List retrieved."),
                )
                logger.info(f"Retrieved {len(projects)} project(s).")
            return projects


async def lsp_start() -> int:
    global DEFAULT_PROJECT_ROOT
    args = get_arg_parser().parse_args()
    if args.version:
        print(__version__)
        return 0

    if args.project_root == "":
        DEFAULT_PROJECT_ROOT = find_project_root(
            ".", ".vectorcode"
        ) or find_project_root(".", ".git")
    else:
        DEFAULT_PROJECT_ROOT = os.path.abspath(args.project_root)

    if DEFAULT_PROJECT_ROOT is None:
        logger.warning("DEFAULT_PROJECT_ROOT is empty.")
    else:
        logger.info(f"{DEFAULT_PROJECT_ROOT=}")

    logger.info("Parsed LSP server CLI arguments: %s", args)
    await asyncio.to_thread(server.start_io)

    return 0


def main():  # pragma: nocover
    config_logging("vectorcode-lsp-server", stdio=False)
    asyncio.run(lsp_start())


if __name__ == "__main__":  # pragma: nocover
    main()
