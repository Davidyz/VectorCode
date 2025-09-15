import argparse
import asyncio
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import shtab

from vectorcode.database import get_database_connector
from vectorcode.database.types import ResultType
from vectorcode.subcommands.vectorise import (
    FilterManager,
    VectoriseStats,
    find_exclude_specs,
    vectorise_worker,
)

try:  # pragma: nocover
    from mcp import ErrorData, McpError
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as e:  # pragma: nocover
    print(
        f"{e.__class__.__name__}:MCP Python SDK not installed. Please install it by installing `vectorcode[mcp]` dependency group.",
        file=sys.stderr,
    )
    sys.exit(1)

from vectorcode.cli_utils import (
    Config,
    SpecResolver,
    config_logging,
    expand_globs,
    expand_path,
    find_project_config_dir,
    get_project_config,
    load_config_file,
)
from vectorcode.subcommands.prompt import prompt_by_categories
from vectorcode.subcommands.query import (
    _prepare_formatted_result,
    get_reranked_results,
    preprocess_query_keywords,
)

logger = logging.getLogger(name=__name__)


@dataclass
class MCPConfig:
    n_results: int = 10
    ls_on_start: bool = False


mcp_config = MCPConfig()


def get_arg_parser():
    parser = argparse.ArgumentParser(prog="vectorcode-mcp-server")
    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=10,
        help="Default number of files to retrieve.",
    )
    parser.add_argument(
        "--ls-on-start",
        action="store_true",
        default=False,
        help="Whether to include the output of `vectorcode ls` in the tool description.",
    )
    shtab.add_argument_to(
        parser,
        ["-s", "--print-completion"],
        parent=parser,
        help="Print completion script.",
    )
    return parser


default_project_root: Optional[str] = None
default_config: Optional[Config] = None


async def list_collections() -> list[str]:
    """
    Returns a list of paths to the projects that have been indexed in the database.
    """

    config = await load_config_file(default_project_root)
    return [i.path for i in await get_database_connector(config).list_collections()]


async def vectorise_files(paths: list[str], project_root: str) -> dict[str, int]:
    logger.info(
        f"vectorise tool called with the following args: {paths=}, {project_root=}"
    )
    project_root = os.path.expanduser(project_root)
    if not os.path.isdir(project_root):
        logger.error(f"Invalid project root: {project_root}")
        raise McpError(
            ErrorData(code=1, message=f"{project_root} is not a valid path.")
        )
    config = await get_project_config(project_root)

    paths = [os.path.expanduser(i) for i in await expand_globs(paths)]
    final_config = await config.merge_from(
        Config(
            files=[i for i in paths if os.path.isfile(i)],
            project_root=project_root,
        )
    )
    filters = FilterManager()
    for ignore_spec_file in find_exclude_specs(final_config):
        if os.path.isfile(ignore_spec_file):
            logger.info(f"Loading ignore specs from {ignore_spec_file}.")
            spec = SpecResolver.from_path(ignore_spec_file)
            filters.add_filter(lambda x: spec.match_file(x, True))

    final_config.files = list(filters(paths))

    database = get_database_connector(final_config)
    try:
        stats = VectoriseStats()
        stats_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(os.cpu_count() or 1)
        tasks = [
            asyncio.create_task(
                vectorise_worker(database, file, semaphore, stats, stats_lock)
            )
            for file in paths
        ]
        for i, task in enumerate(asyncio.as_completed(tasks), start=1):
            await task

        await database.check_orphanes()

        return stats.to_dict()
    except Exception as e:  # pragma: nocover
        if isinstance(e, McpError):
            logger.error("Failed to access collection at %s", project_root)
            raise
        else:
            raise McpError(
                ErrorData(
                    code=1,
                    message="\n".join(traceback.format_exception(e)),
                )
            ) from e


async def query_tool(
    n_query: int, query_messages: list[str], project_root: str
) -> list[str]:
    """
    n_query: number of files to retrieve;
    query_messages: keywords to query.
    collection_path: Directory to the repository;
    """
    logger.info(
        f"query tool called with the following args: {n_query=}, {query_messages=}, {project_root=}"
    )
    project_root = os.path.expanduser(project_root)
    if not os.path.isdir(project_root):
        logger.error("Invalid project root: %s", project_root)
        raise McpError(
            ErrorData(
                code=1,
                message="Use `list_collections` tool to get a list of valid paths for this field.",
            )
        )

    config = await get_project_config(project_root)
    preprocess_query_keywords(config)
    config.n_result = n_query

    try:
        database = get_database_connector(config)
        reranked_results = await get_reranked_results(config, database)
        return list(str(i) for i in _prepare_formatted_result(reranked_results))

    except Exception as e:  # pragma: nocover
        if isinstance(e, McpError):
            logger.error("Failed to access collection at %s", project_root)
            raise
        else:
            raise McpError(
                ErrorData(
                    code=1,
                    message="\n".join(traceback.format_exception(e)),
                )
            ) from e


async def ls_files(project_root: str) -> list[str]:
    """
    project_root: Directory to the repository. MUST be from the vectorcode `ls` tool or user input;
    """
    configs = await get_project_config(expand_path(project_root, True))
    database = get_database_connector(configs)
    return list(
        i.path
        for i in (
            await database.list_collection_content(what=ResultType.document)
        ).files
    )


async def rm_files(files: list[str], project_root: str):
    """
    files: list of paths of the files to be removed;
    project_root: Directory to the repository. MUST be from the vectorcode `ls` tool or user input;
    """
    configs = await get_project_config(expand_path(project_root, True))
    configs.rm_paths = [str(expand_path(i, True)) for i in files if os.path.isfile(i)]

    if configs.rm_paths:
        database = get_database_connector(configs)
        num_deleted = await database.delete()
        return f"Removed {num_deleted} files from the database of the project located at {project_root}"
    else:
        logger.warning(f"The provided paths were invalid: {configs.rm_paths}")


async def mcp_server():
    global default_config, default_project_root

    local_config_dir = await find_project_config_dir(".")

    default_instructions = "\n".join(
        "\n".join(i) for i in prompt_by_categories.values()
    )
    if local_config_dir is not None:
        logger.info("Found project config: %s", local_config_dir)
        project_root = str(Path(local_config_dir).parent.resolve())

        default_project_root = project_root
        default_config = await get_project_config(project_root)
        default_config.project_root = project_root
        if mcp_config.ls_on_start:
            logger.info("Adding available collections to the server instructions.")
            default_instructions += "\nYou have access to the following collections:\n"
            for name in await list_collections():
                default_instructions += f"<collection>{name}</collection>"

    mcp = FastMCP("VectorCode", instructions=default_instructions)
    mcp.add_tool(
        fn=list_collections,
        name="ls",
        description="\n".join(
            prompt_by_categories["ls"] + prompt_by_categories["general"]
        ),
    )

    mcp.add_tool(
        fn=query_tool,
        name="query",
        description="\n".join(
            prompt_by_categories["query"] + prompt_by_categories["general"]
        ),
    )

    mcp.add_tool(
        fn=vectorise_files,
        name="vectorise",
        description="\n".join(
            prompt_by_categories["vectorise"] + prompt_by_categories["general"]
        ),
    )

    mcp.add_tool(
        fn=rm_files,
        name="files_rm",
        description="Remove files from VectorCode embedding database.",
    )

    mcp.add_tool(
        fn=ls_files,
        name="files_ls",
        description="List files that have been indexed by VectorCode.",
    )

    return mcp


def parse_cli_args(args: Optional[list[str]] = None) -> MCPConfig:
    parser = get_arg_parser()
    parsed_args = parser.parse_args(args or sys.argv[1:])
    return MCPConfig(n_results=parsed_args.number, ls_on_start=parsed_args.ls_on_start)


async def run_server():  # pragma: nocover
    try:
        mcp = await mcp_server()
        await mcp.run_stdio_async()
    finally:
        return 0


def main():  # pragma: nocover
    global mcp_config
    config_logging("vectorcode-mcp-server", stdio=False)
    mcp_config = parse_cli_args()
    assert mcp_config.n_results > 0 and mcp_config.n_results % 1 == 0, (
        "--number must be used with a positive integer!"
    )
    return asyncio.run(run_server())


if __name__ == "__main__":  # pragma: nocover
    main()
