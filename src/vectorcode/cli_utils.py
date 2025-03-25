import argparse
import glob
import json
import os
from dataclasses import dataclass, field, fields
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import shtab

from vectorcode import __version__

PathLike = Union[str, Path]

GLOBAL_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), ".config", "vectorcode", "config.json"
)

CHECK_OPTIONS = ["config"]


class QueryInclude(StrEnum):
    path = "path"
    document = "document"

    def to_header(self) -> str:
        """
        Make the string into a nice-looking format for printing in the terminal.
        """
        if self.value == "document":
            return f"{self.value.capitalize()}:\n"
        return f"{self.value.capitalize()}: "


class CliAction(Enum):
    vectorise = "vectorise"
    query = "query"
    drop = "drop"
    ls = "ls"
    init = "init"
    version = "version"
    check = "check"
    update = "update"
    clean = "clean"
    prompts = "prompts"
    chunks = "chunks"


@dataclass
class Config:
    no_stderr: bool = False
    recursive: bool = False
    to_be_deleted: list[str] = field(default_factory=list)
    pipe: bool = False
    action: Optional[CliAction] = None
    files: list[PathLike] = field(default_factory=list)
    project_root: Optional[PathLike] = None
    query: Optional[list[str]] = None
    host: str = "127.0.0.1"
    port: int = 8000
    embedding_function: str = "SentenceTransformerEmbeddingFunction"  # This should fallback to whatever the default is.
    embedding_params: dict[str, Any] = field(default_factory=(lambda: {}))
    n_result: int = 1
    force: bool = False
    db_path: Optional[str] = "~/.local/share/vectorcode/chromadb/"
    db_settings: Optional[dict] = None
    chunk_size: int = 2500
    overlap_ratio: float = 0.2
    query_multiplier: int = -1
    query_exclude: list[PathLike] = field(default_factory=list)
    reranker: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_params: dict[str, Any] = field(default_factory=dict)
    check_item: Optional[str] = None
    use_absolute_path: bool = False
    include: list[QueryInclude] = field(
        default_factory=lambda: [QueryInclude.path, QueryInclude.document]
    )
    hnsw: dict[str, str | int] = field(default_factory=dict)
    chunk_filters: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    async def import_from(cls, config_dict: dict[str, Any]) -> "Config":
        """
        Raise IOError if db_path is not valid.
        """
        db_path = config_dict.get("db_path")
        host = config_dict.get("host") or "localhost"
        port = config_dict.get("port") or 8000
        if db_path is None:
            db_path = os.path.expanduser("~/.local/share/vectorcode/chromadb/")
        elif not os.path.isdir(db_path):
            raise IOError(
                f"The configured db_path ({str(db_path)}) is not a valid directory."
            )
        return Config(
            **{
                "embedding_function": config_dict.get(
                    "embedding_function", "SentenceTransformerEmbeddingFunction"
                ),
                "embedding_params": config_dict.get("embedding_params", {}),
                "host": host,
                "port": port,
                "db_path": db_path,
                "chunk_size": config_dict.get("chunk_size", 2500),
                "overlap_ratio": config_dict.get("overlap_ratio", 0.2),
                "query_multiplier": config_dict.get("query_multiplier", -1),
                "reranker": config_dict.get(
                    "reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
                "reranker_params": config_dict.get("reranker_params", {}),
                "db_settings": config_dict.get("db_settings", None),
                "hnsw": config_dict.get("hnsw", {}),
                "chunk_filters": config_dict.get("chunk_filters", {}),
            }
        )

    async def merge_from(self, other: "Config") -> "Config":
        """Return the merged config."""
        final_config = {}
        default_config = Config()
        for merged_field in fields(self):
            field_name = merged_field.name

            other_val = getattr(other, field_name)
            self_val = getattr(self, field_name)
            if isinstance(other_val, dict) and isinstance(self_val, dict):
                final_config[field_name] = {}
                final_config[field_name].update(self_val)
                final_config[field_name].update(other_val)
            else:
                final_config[field_name] = other_val
                if not final_config[field_name] or final_config[field_name] == getattr(
                    default_config, field_name
                ):
                    final_config[field_name] = self_val
        return Config(**final_config)


def get_cli_parser():
    shared_parser = argparse.ArgumentParser(add_help=False)
    chunking_parser = argparse.ArgumentParser(add_help=False)
    chunking_parser.add_argument(
        "--overlap", "-o", type=float, help="Ratio of overlaps between chunks."
    )
    chunking_parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=-1,
        help="Size of chunks (-1 for no chunking).",
    )
    shared_parser.add_argument(
        "--project_root",
        default=None,
        help="Project root to be used as an identifier of the project.",
    ).complete = shtab.DIRECTORY
    shared_parser.add_argument(
        "--pipe",
        "-p",
        action="store_true",
        default=False,
        help="Print structured output for other programs to process.",
    )
    shared_parser.add_argument(
        "--no_stderr",
        action="store_true",
        default=False,
        help="Supress all STDERR messages.",
    )
    main_parser = argparse.ArgumentParser(
        "vectorcode",
        parents=[shared_parser],
        description=f"VectorCode {__version__}: A CLI RAG utility.",
    )
    shtab.add_argument_to(
        main_parser,
        ["-s", "--print-completion"],
        parent=main_parser,
        help="Print completion script.",
    )
    subparsers = main_parser.add_subparsers(
        dest="action",
        required=False,
        title="subcommands",
    )
    subparsers.add_parser("ls", parents=[shared_parser], help="List all collections.")

    vectorise_parser = subparsers.add_parser(
        "vectorise",
        parents=[shared_parser, chunking_parser],
        help="Vectorise and send documents to chromadb.",
    )
    vectorise_parser.add_argument(
        "file_paths", nargs="*", help="Paths to files to be vectorised."
    ).complete = shtab.FILE
    vectorise_parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=False,
        help="Recursive indexing for directories.",
    )
    vectorise_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Force to vectorise the file(s) against the gitignore.",
    )

    query_parser = subparsers.add_parser(
        "query",
        parents=[shared_parser, chunking_parser],
        help="Send query to retrieve documents.",
    )
    query_parser.add_argument("query", nargs="+", help="Query keywords.")
    query_parser.add_argument(
        "--multiplier", "-m", type=int, default=-1, help="Query multiplier."
    )
    query_parser.add_argument(
        "-n", "--number", type=int, default=1, help="Number of results to retrieve."
    )
    query_parser.add_argument(
        "--exclude", nargs="*", help="Files to exclude from query results."
    ).complete = shtab.FILE
    query_parser.add_argument(
        "--absolute",
        default=False,
        action="store_true",
        help="Use absolute path when returning the retrieval results.",
    )
    query_parser.add_argument(
        "--include",
        choices=list(i.value for i in QueryInclude),
        nargs="+",
        help="What to include in the final output.",
        default=["path", "document"],
    )

    subparsers.add_parser("drop", parents=[shared_parser], help="Remove a collection.")

    init_parser = subparsers.add_parser(
        "init",
        parents=[shared_parser],
        help="Initialise a directory as VectorCode project root.",
    )
    init_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Wipe current project config and overwrite with global config (if it exists).",
    )

    subparsers.add_parser(
        "version", parents=[shared_parser], help="Print the version number."
    )
    check_parser = subparsers.add_parser(
        "check", parents=[shared_parser], help="Check for project-local setup."
    )

    check_parser.add_argument(
        "check_item",
        choices=CHECK_OPTIONS,
        type=str,
        help=f"Item to be checked. Possible options: [{', '.join(CHECK_OPTIONS)}]",
    )

    subparsers.add_parser(
        "update",
        parents=[shared_parser],
        help="Update embeddings in the database for indexed files.",
    )

    subparsers.add_parser(
        "clean",
        parents=[shared_parser],
        help="Remove empty collections in the database.",
    )

    subparsers.add_parser(
        "prompts",
        parents=[shared_parser],
        help="Print a list of guidelines intended to be used as system prompts for an LLM.",
    )

    chunks_parser = subparsers.add_parser(
        "chunks",
        parents=[shared_parser, chunking_parser],
        help="Print a JSON array containing chunked text.",
    )
    chunks_parser.add_argument(
        "file_paths", nargs="*", help="Paths to files to be chunked."
    ).complete = shtab.FILE
    return main_parser


async def parse_cli_args(args: Optional[Sequence[str]] = None):
    main_parser = get_cli_parser()
    main_args = main_parser.parse_args(args)
    if main_args.action is None:
        main_args = main_parser.parse_args(["--help"])

    files = []
    query = None
    recursive = False
    number_of_result = 1
    force = False
    chunk_size = -1
    overlap_ratio = 0.2
    query_multiplier = -1
    query_exclude = []
    query_include = ["path", "document"]
    check_item = None
    absolute = False
    match main_args.action:
        case "vectorise":
            files = main_args.file_paths
            recursive = main_args.recursive
            force = main_args.force
            chunk_size = main_args.chunk_size
            overlap_ratio = main_args.overlap
        case "query":
            query = main_args.query
            number_of_result = main_args.number
            query_multiplier = main_args.multiplier
            query_exclude = main_args.exclude
            absolute = main_args.absolute
            query_include = main_args.include
        case "check":
            check_item = main_args.check_item
        case "init":
            force = main_args.force
        case "chunks":
            files = main_args.file_paths
            chunk_size = main_args.chunk_size
            overlap_ratio = main_args.overlap
    return Config(
        no_stderr=main_args.no_stderr,
        action=CliAction(main_args.action),
        files=files,
        project_root=main_args.project_root,
        query=query,
        recursive=recursive,
        n_result=number_of_result,
        pipe=main_args.pipe,
        force=force,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
        query_multiplier=query_multiplier,
        query_exclude=query_exclude,
        check_item=check_item,
        use_absolute_path=absolute,
        include=[QueryInclude(i) for i in query_include],
    )


def expand_envs_in_dict(d: dict):
    if not isinstance(d, dict):
        return
    stack = [d]
    while stack:
        curr = stack.pop()
        for k in curr.keys():
            if isinstance(curr[k], str):
                curr[k] = os.path.expandvars(curr[k])
            elif isinstance(curr[k], dict):
                stack.append(curr[k])


async def load_config_file(path: Optional[PathLike] = None):
    """Load config file from ~/.config/vectorcode/config.json"""
    if path is None:
        path = GLOBAL_CONFIG_PATH
    if os.path.isfile(path):
        with open(path) as fin:
            config = json.load(fin)
        expand_envs_in_dict(config)
        return await Config.import_from(config)
    return Config()


async def find_project_config_dir(start_from: PathLike = "."):
    """Returns the project-local config directory."""
    current_dir = Path(start_from).resolve()
    project_root_anchors = [".vectorcode", ".git"]
    while current_dir:
        for anchor in project_root_anchors:
            to_be_checked = os.path.join(current_dir, anchor)
            if os.path.isdir(to_be_checked):
                return to_be_checked
        parent = current_dir.parent
        if parent.resolve() == current_dir:
            return
        current_dir = parent.resolve()


def find_project_root(
    start_from: PathLike, root_anchor: PathLike = ".vectorcode"
) -> str | None:
    start_from = Path(start_from)
    if os.path.isfile(start_from):
        start_from = start_from.parent

    while start_from:
        if (start_from / Path(root_anchor)).is_dir():
            return str(start_from.absolute())
        if start_from == start_from.parent:
            return
        start_from = start_from.parent


async def get_project_config(project_root: PathLike) -> Config:
    """
    Load config file for `project_root`.
    Fallback to global config, and then default config.
    """
    if not os.path.isabs(project_root):
        project_root = os.path.abspath(project_root)
    local_config_path = os.path.join(project_root, ".vectorcode", "config.json")
    if os.path.isfile(os.path.join(project_root, ".vectorcode", "config.json")):
        config = await load_config_file(local_config_path)
    else:
        config = await load_config_file()
    config.project_root = project_root
    return config


def expand_path(path: PathLike, absolute: bool = False) -> PathLike:
    expanded = os.path.expanduser(os.path.expandvars(path))
    if absolute:
        return os.path.abspath(expanded)
    return expanded


async def expand_globs(
    paths: list[PathLike], recursive: bool = False
) -> list[PathLike]:
    result = set()
    stack = paths
    while stack:
        curr = stack.pop()
        if os.path.isfile(curr):
            result.add(expand_path(curr))
        elif "*" in str(curr):
            stack.extend(glob.glob(str(curr), recursive=recursive))
        elif os.path.isdir(curr) and recursive:
            stack.extend(glob.glob(os.path.join(curr, "**", "*"), recursive=recursive))
    return list(result)
