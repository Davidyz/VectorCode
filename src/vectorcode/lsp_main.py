import argparse
import asyncio
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
    find_project_root,
    load_config_file,
    parse_cli_args,
)
from vectorcode.common import get_client, get_collection, try_server
from vectorcode.subcommands.ls import get_collection_list
from vectorcode.subcommands.query import get_query_result_files

cached_project_configs: dict[str, Config] = {}
DEFAULT_PROJECT_ROOT: str | None = None


async def make_caches(project_root: str):
    assert os.path.isabs(project_root)
    if cached_project_configs.get(project_root) is None:
        config_file = os.path.join(project_root, ".vectorcode", "config.json")
        if not os.path.isfile(config_file):
            config_file = None
        cached_project_configs[project_root] = await load_config_file(config_file)
    config = cached_project_configs[project_root]
    config.project_root = project_root
    host, port = config.host, config.port
    if not await try_server(host, port):  # pragma: nocover
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
    parsed_args = await parse_cli_args(args)
    if parsed_args.action not in {CliAction.query, CliAction.ls}:
        print(
            f"Unsupported vectorcode subcommand: {str(parsed_args.action)}",
            file=sys.stderr,
        )
        return
    if parsed_args.project_root is None:
        assert DEFAULT_PROJECT_ROOT is not None, (
            "Failed to automatically resolve project root!"
        )

        parsed_args.project_root = DEFAULT_PROJECT_ROOT
    elif DEFAULT_PROJECT_ROOT is None:
        DEFAULT_PROJECT_ROOT = str(parsed_args.project_root)

    parsed_args.project_root = os.path.abspath(str(parsed_args.project_root))
    await make_caches(parsed_args.project_root)
    final_configs = await cached_project_configs[parsed_args.project_root].merge_from(
        parsed_args
    )
    final_configs.pipe = True
    progress_token = str(uuid.uuid4())
    client = await get_client(final_configs)
    collection = await get_collection(
        client=client,
        configs=final_configs,
        make_if_missing=final_configs.action in {CliAction.vectorise},
    )
    await ls.progress.create_async(progress_token)
    match final_configs.action:
        case CliAction.query:
            ls.progress.begin(
                progress_token,
                types.WorkDoneProgressBegin(
                    "VectorCode",
                    message="Retrieving from VectorCode",
                ),
            )
            final_results = []
            for path in await get_query_result_files(
                collection=collection,
                configs=final_configs,
            ):
                if os.path.isfile(path):
                    with open(path) as fin:
                        output_path = path
                        if not final_configs.use_absolute_path:
                            output_path = os.path.relpath(
                                path, final_configs.project_root
                            )
                        final_results.append(
                            {"path": output_path, "document": fin.read()}
                        )
            ls.progress.end(
                progress_token,
                types.WorkDoneProgressEnd(
                    message=f"Retrieved {len(final_results)} result{'s' if len(final_results) > 1 else ''} in {round(time.time() - start_time, 2)}s."
                ),
            )
            return final_results
        case CliAction.ls:
            ls.progress.begin(
                progress_token,
                types.WorkDoneProgressBegin(
                    "VectorCode",
                    message="Looking for other projects indexed by VectorCode",
                ),
            )
            projects: list[dict] = await get_collection_list(client)

            ls.progress.end(
                progress_token,
                types.WorkDoneProgressEnd(message="List retrieved."),
            )
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

    await asyncio.to_thread(server.start_io)

    return 0


def main():  # pragma: nocover
    asyncio.run(lsp_start())


if __name__ == "__main__":  # pragma: nocover
    main()
