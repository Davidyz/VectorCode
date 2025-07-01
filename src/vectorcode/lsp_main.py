import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
import uuid
from typing import cast

import shtab
from chromadb.types import Where

from vectorcode.subcommands.vectorise import (
    VectoriseStats,
    chunked_add,
    exclude_paths_by_spec,
    find_exclude_specs,
    load_files_from_include,
    remove_orphanes,
)

try:  # pragma: nocover
    from lsprotocol import types
    from pygls.exceptions import (
        JsonRpcException,
        JsonRpcInternalError,
        JsonRpcInvalidRequest,
    )
    from pygls.server import LanguageServer
except ModuleNotFoundError as e:  # pragma: nocover
    print(
        f"{e.__class__.__name__}: Please install the `vectorcode[lsp]` dependency group to use the LSP feature.",
        file=sys.stderr,
    )
    sys.exit(1)
from chromadb.errors import InvalidCollectionException

from vectorcode import __version__
from vectorcode.cli_utils import (
    CliAction,
    FilesAction,
    cleanup_path,
    config_logging,
    expand_globs,
    expand_path,
    find_project_root,
    get_project_config,
    parse_cli_args,
)
from vectorcode.common import ClientManager, get_collection, list_collection_files
from vectorcode.subcommands.ls import get_collection_list
from vectorcode.subcommands.query import build_query_results

DEFAULT_PROJECT_ROOT: str | None = None
logger = logging.getLogger(__name__)


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
    shtab.add_argument_to(
        parser,
        ["-s", "--print-completion"],
        parent=parser,
        help="Print completion script.",
    )
    return parser


server: LanguageServer = LanguageServer(name="vectorcode-server", version=__version__)


@server.command("vectorcode")
async def execute_command(ls: LanguageServer, args: list[str]):
    try:
        global DEFAULT_PROJECT_ROOT
        start_time = time.time()
        logger.info("Received command arguments: %s", args)
        parsed_args = await parse_cli_args(args)
        logger.info("Parsed command arguments: %s", parsed_args)
        if parsed_args.project_root is None:
            if DEFAULT_PROJECT_ROOT is not None:
                parsed_args.project_root = DEFAULT_PROJECT_ROOT
                logger.warning("Using DEFAULT_PROJECT_ROOT: %s", DEFAULT_PROJECT_ROOT)
        elif DEFAULT_PROJECT_ROOT is None:
            logger.warning(
                "Updating DEFAULT_PROJECT_ROOT to %s", parsed_args.project_root
            )
            DEFAULT_PROJECT_ROOT = str(parsed_args.project_root)

        collection = None
        if parsed_args.project_root is not None:
            parsed_args.project_root = os.path.abspath(str(parsed_args.project_root))

            final_configs = await (
                await get_project_config(parsed_args.project_root)
            ).merge_from(parsed_args)
            final_configs.pipe = True
        else:
            final_configs = parsed_args
        logger.info("Merged final configs: %s", final_configs)
        async with ClientManager().get_client(final_configs) as client:
            progress_token = str(uuid.uuid4())

            if final_configs.action in {
                CliAction.vectorise,
                CliAction.query,
                CliAction.files,
            }:
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
                            message=f"Querying {cleanup_path(str(final_configs.project_root))}",
                        ),
                    )
                    final_results = []
                    try:
                        assert collection is not None, (
                            "Failed to find the correct collection."
                        )
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
                            message="Looking for available projects indexed by VectorCode",
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
                case CliAction.vectorise:
                    assert collection is not None, (
                        "Failed to find the correct collection."
                    )
                    ls.progress.begin(
                        progress_token,
                        types.WorkDoneProgressBegin(
                            title="VectorCode",
                            message="Vectorising files...",
                            percentage=0,
                        ),
                    )
                    files = await expand_globs(
                        final_configs.files
                        or load_files_from_include(str(final_configs.project_root)),
                        recursive=final_configs.recursive,
                        include_hidden=final_configs.include_hidden,
                    )
                    if not final_configs.force:  # pragma: nocover
                        # tested in 'vectorise.py'
                        for spec in find_exclude_specs(final_configs):
                            if os.path.isfile(spec):
                                logger.info(f"Loading ignore specs from {spec}.")
                                files = exclude_paths_by_spec(
                                    (str(i) for i in files), spec
                                )
                    stats = VectoriseStats()
                    collection_lock = asyncio.Lock()
                    stats_lock = asyncio.Lock()
                    max_batch_size = await client.get_max_batch_size()
                    semaphore = asyncio.Semaphore(os.cpu_count() or 1)
                    tasks = [
                        asyncio.create_task(
                            chunked_add(
                                str(file),
                                collection,
                                collection_lock,
                                stats,
                                stats_lock,
                                final_configs,
                                max_batch_size,
                                semaphore,
                            )
                        )
                        for file in files
                    ]
                    for i, task in enumerate(asyncio.as_completed(tasks), start=1):
                        await task
                        ls.progress.report(
                            progress_token,
                            types.WorkDoneProgressReport(
                                message="Vectorising files...",
                                percentage=int(100 * i / len(tasks)),
                            ),
                        )

                    await remove_orphanes(
                        collection, collection_lock, stats, stats_lock
                    )

                    ls.progress.end(
                        progress_token,
                        types.WorkDoneProgressEnd(
                            message=f"Vectorised {stats.add + stats.update} files."
                        ),
                    )
                    return stats.to_dict()
                case CliAction.files:
                    if collection is None:  # pragma: nocover
                        raise InvalidCollectionException(
                            f"Failed to find the corresponding collection for {final_configs.project_root}"
                        )
                    match final_configs.files_action:
                        case FilesAction.ls:
                            return await list_collection_files(collection)
                        case FilesAction.rm:
                            to_be_removed = list(
                                str(expand_path(p, True))
                                for p in final_configs.rm_paths
                                if os.path.isfile(p)
                            )
                            if len(to_be_removed) == 0:
                                return
                            ls.progress.begin(
                                progress_token,
                                types.WorkDoneProgressBegin(
                                    title="VectorCode",
                                    message=f"Removing {len(to_be_removed)} file(s).",
                                ),
                            )
                            await collection.delete(
                                where=cast(
                                    Where,
                                    {"path": {"$in": to_be_removed}},
                                )
                            )
                            ls.progress.end(
                                progress_token,
                                types.WorkDoneProgressEnd(
                                    message="Removal finished.",
                                ),
                            )
                case _ as c:  # pragma: nocover
                    error_message = f"Unsupported vectorcode subcommand: {str(c)}"
                    logger.error(
                        error_message,
                    )
                    raise JsonRpcInvalidRequest(error_message)
    except Exception as e:  # pragma: nocover
        if isinstance(e, JsonRpcException):
            # pygls exception. raise it as is.
            raise
        else:
            # wrap non-pygls errors for error codes.
            raise JsonRpcInternalError(message=traceback.format_exc()) from e


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
    try:
        await asyncio.to_thread(server.start_io)
    finally:
        await ClientManager().kill_servers()
        return 0


def main():  # pragma: nocover
    config_logging("vectorcode-lsp-server", stdio=False)
    asyncio.run(lsp_start())


if __name__ == "__main__":  # pragma: nocover
    main()
