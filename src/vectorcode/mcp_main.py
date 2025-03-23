import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.errors import InvalidCollectionException

try:  # pragma: nocover
    from mcp import ErrorData, McpError
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: nocover
    print(
        "MCP Python SDK not installed. Please install it by installing `vectorcode[mcp]` dependency group.",
        file=sys.stderr,
    )
    sys.exit(1)

from vectorcode.cli_utils import (
    Config,
    find_project_config_dir,
    get_project_config,
    load_config_file,
)
from vectorcode.common import get_client, get_collection, get_collections
from vectorcode.subcommands.prompt import prompt_strings
from vectorcode.subcommands.query import get_query_result_files

mcp = FastMCP("VectorCode", instructions="\n".join(prompt_strings))


default_config: Optional[Config] = None
default_client: Optional[AsyncClientAPI] = None
default_collection: Optional[AsyncCollection] = None


async def list_collections() -> list[str]:
    global default_config, default_client, default_collection
    names: list[str] = []
    client = default_client
    if client is None:
        # load from global config when failed to detect a project-local config.
        client = await get_client(await load_config_file())
    async for col in get_collections(client):
        if col.metadata is not None:
            names.append(str(col.metadata.get("path")))
    return names


async def query_tool(
    n_query: int, query_messages: list[str], project_root: str
) -> list[str]:
    """
    n_query: number of files to retrieve;
    query_messages: keywords to query.
    collection_path: Directory to the repository;
    """
    if not os.path.isdir(project_root):
        raise McpError(
            ErrorData(
                code=1,
                message="Use `list_collections` tool to get a list of valid paths for this field.",
            )
        )
    else:
        config = await get_project_config(project_root)
        try:
            client = await get_client(config)
            collection = await get_collection(client, config, False)
        except Exception:
            raise McpError(
                ErrorData(
                    code=1,
                    message=f"Failed to access the collection at {project_root}. Use `list_collections` tool to get a list of valid paths for this field.",
                )
            )
    if collection is None:
        raise McpError(
            ErrorData(
                code=1,
                message=f"Failed to access the collection at {project_root}. Use `list_collections` tool to get a list of valid paths for this field.",
            )
        )
    query_config = await config.merge_from(
        Config(n_result=n_query, query=query_messages)
    )
    result_paths = await get_query_result_files(
        collection=collection,
        configs=query_config,
    )
    results: list[str] = []
    for path in result_paths:
        if os.path.isfile(path):
            with open(path) as fin:
                rel_path = os.path.relpath(path, config.project_root)
                results.append(
                    f"<path>{rel_path}</path>\n<content>{fin.read()}</content>",
                )

    return results


async def mcp_server():
    global default_config, default_client, default_collection
    # sys.stderr = open(os.devnull, "w")
    local_config_dir = await find_project_config_dir(".")
    print(local_config_dir, file=sys.stderr)

    if local_config_dir is not None:
        project_root = str(Path(local_config_dir).parent.resolve())

        default_config = await load_config_file(
            os.path.join(project_root, ".vectorcode", "config.json")
        )
        default_config.project_root = project_root
        default_client = await get_client(default_config)
        try:
            default_collection = await get_collection(default_client, default_config)
        except InvalidCollectionException:
            default_collection = None

    mcp.add_tool(
        fn=list_collections,
        name="ls",
        description="List all projects indexed by VectorCode. Call this before making queries.",
    )

    mcp.add_tool(
        fn=query_tool,
        name="query",
        description="Use VectorCode to perform vector similarity search on repositories and return a list of relevant file paths and contents. Make sure `project_root` is one of the values from the `list_collections` tool.",
    )

    return mcp


async def run_server():
    mcp = await mcp_server()
    await mcp.run_stdio_async()
    return 0


def main():
    return asyncio.run(run_server())


if __name__ == "__main__":  # pragma: nocover
    main()
