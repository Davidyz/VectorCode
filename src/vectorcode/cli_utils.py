import argparse
import atexit
import glob
import logging
import os
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Sequence, Union

import json5
import shtab
from filelock import AsyncFileLock
from pathspec import GitIgnoreSpec

from vectorcode import __version__

logger = logging.getLogger(name=__name__)


GLOBAL_CONFIG_DIR = os.path.join(
    os.path.expanduser("~"),
    ".config",
    "vectorcode",
)
GLOBAL_INCLUDE_SPEC = os.path.join(
    os.path.expanduser("~"), ".config", "vectorcode", "vectorcode.include"
)
GLOBAL_EXCLUDE_SPEC = os.path.join(
    os.path.expanduser("~"), ".config", "vectorcode", "vectorcode.exclude"
)

CHECK_OPTIONS = ["config"]


class QueryInclude(StrEnum):
    path = "path"
    document = "document"
    chunk = "chunk"

    def to_header(self) -> str:
        """
        Make the string into a nice-looking format for printing in the terminal.
        """
        if self.value == "document":
            return f"{self.value.capitalize()}:\n"
        return f"{self.value.capitalize()}: "


class PromptCategory(StrEnum):
    query = "query"
    vectorise = "vectorise"
    ls = "ls"


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
    files = "files"


class FilesAction(StrEnum):
    ls = "ls"
    rm = "rm"


@dataclass
class Config:
    no_stderr: bool = False
    recursive: bool = False
    include_hidden: bool = False
    to_be_deleted: list[str] = field(default_factory=list)
    pipe: bool = False
    action: Optional[CliAction] = None
    files: list[Union[str, os.PathLike]] = field(default_factory=list)
    project_root: Optional[Union[str, Path]] = None
    query: Optional[list[str]] = None
    db_url: str = "http://127.0.0.1:8000"
    embedding_function: str = "SentenceTransformerEmbeddingFunction"  # This should fallback to whatever the default is.
    embedding_params: dict[str, Any] = field(default_factory=(lambda: {}))
    n_result: int = 1
    force: bool = False
    db_path: Optional[str] = "~/.local/share/vectorcode/chromadb/"
    db_log_path: str = "~/.local/share/vectorcode/"
    db_settings: Optional[dict] = None
    chunk_size: int = 2500
    overlap_ratio: float = 0.2
    query_multiplier: int = -1
    query_exclude: list[Union[str, os.PathLike]] = field(default_factory=list)
    reranker: Optional[str] = "CrossEncoderReranker"
    reranker_params: dict[str, Any] = field(default_factory=lambda: {})
    check_item: Optional[str] = None
    use_absolute_path: bool = False
    include: list[QueryInclude] = field(
        default_factory=lambda: [QueryInclude.path, QueryInclude.document]
    )
    hnsw: dict[str, str | int] = field(default_factory=dict)
    chunk_filters: dict[str, list[str]] = field(default_factory=dict)
    filetype_map: dict[str, list[str]] = field(default_factory=dict)
    encoding: str = "utf8"
    hooks: bool = False
    prompt_categories: Optional[list[str]] = None
    files_action: Optional[FilesAction] = None
    rm_paths: list[str] = field(default_factory=list)

    @classmethod
    async def import_from(cls, config_dict: dict[str, Any]) -> "Config":
        """
        Raise IOError if db_path is not valid.
        """
        default_config = Config()
        db_path = config_dict.get("db_path")

        if db_path is None:
            db_path = os.path.expanduser("~/.local/share/vectorcode/chromadb/")
        elif not os.path.isdir(db_path):
            raise IOError(
                f"The configured db_path ({str(db_path)}) is not a valid directory."
            )
        return Config(
            **{
                "embedding_function": config_dict.get(
                    "embedding_function", default_config.embedding_function
                ),
                "embedding_params": config_dict.get(
                    "embedding_params", default_config.embedding_params
                ),
                "db_url": config_dict.get("db_url", default_config.db_url),
                "db_path": db_path,
                "db_log_path": os.path.expanduser(
                    config_dict.get("db_log_path", default_config.db_log_path)
                ),
                "chunk_size": config_dict.get("chunk_size", default_config.chunk_size),
                "overlap_ratio": config_dict.get(
                    "overlap_ratio", default_config.overlap_ratio
                ),
                "query_multiplier": config_dict.get(
                    "query_multiplier", default_config.query_multiplier
                ),
                "reranker": config_dict.get("reranker", default_config.reranker),
                "reranker_params": config_dict.get(
                    "reranker_params", default_config.reranker_params
                ),
                "db_settings": config_dict.get(
                    "db_settings", default_config.db_settings
                ),
                "hnsw": config_dict.get("hnsw", default_config.hnsw),
                "chunk_filters": config_dict.get(
                    "chunk_filters", default_config.chunk_filters
                ),
                "filetype_map": config_dict.get(
                    "filetype_map", default_config.filetype_map
                ),
                "encoding": config_dict.get("encoding", default_config.encoding),
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
    __default_config = Config()
    shared_parser = argparse.ArgumentParser(add_help=False)
    chunking_parser = argparse.ArgumentParser(add_help=False)
    chunking_parser.add_argument(
        "--overlap",
        "-o",
        type=float,
        default=__default_config.overlap_ratio,
        help="Ratio of overlaps between chunks.",
    )
    chunking_parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=__default_config.chunk_size,
        help="Size of chunks (-1 for no chunking).",
    )
    chunking_parser.add_argument(
        "--encoding",
        type=str,
        default=__default_config.encoding,
        help="Encoding used by the files. See https://docs.python.org/3/library/codecs.html#standard-encodings for supported encodings. Use `_auto` for automatic encoding detection.",
    )
    shared_parser.add_argument(
        "--project_root",
        default=None,
        help="Project root to be used as an identifier of the project.",
    ).complete = shtab.DIRECTORY  # type:ignore
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
        help="Suppress all STDERR messages.",
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
    ).complete = shtab.FILE  # type:ignore
    vectorise_parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=False,
        help="Recursive indexing for directories.",
    )
    vectorise_parser.add_argument(
        "--include-hidden",
        action="store_true",
        default=False,
        help="Include hidden files.",
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
        "--multiplier",
        "-m",
        type=int,
        default=__default_config.query_multiplier,
        help="Query multiplier.",
    )
    query_parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=__default_config.n_result,
        help="Number of results to retrieve.",
    )
    query_parser.add_argument(
        "--exclude", nargs="*", help="Files to exclude from query results."
    ).complete = shtab.FILE  # type:ignore
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
        default=__default_config.include,
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
    init_parser.add_argument(
        "--hooks",
        action="store_true",
        default=False,
        help="Add git hooks to the current project, if it's a git repo.",
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

    prompts_parser = subparsers.add_parser(
        "prompts",
        parents=[shared_parser],
        help="Print a list of guidelines intended to be used as system prompts for an LLM.",
    )
    prompts_parser.add_argument(
        "prompt_categories",
        choices=[str(i) for i in PromptCategory],
        type=PromptCategory,
        nargs="*",
        help="The subcommand(s) to get the prompts for. When not provided, VectorCode will print the prompts for `query`.",
        default=None,
    )

    chunks_parser = subparsers.add_parser(
        "chunks",
        parents=[shared_parser, chunking_parser],
        help="Print a JSON array containing chunked text.",
    )
    chunks_parser.add_argument(
        "file_paths", nargs="*", help="Paths to files to be chunked."
    ).complete = shtab.FILE  # type:ignore

    files_parser = subparsers.add_parser(
        "files", parents=[shared_parser], help="Manipulate files from a collection."
    )
    files_subparser = files_parser.add_subparsers(
        dest="files_action", required=True, title="Collecton file operations"
    )
    files_subparser.add_parser(
        "ls", parents=[shared_parser], help="List files in the collection."
    )
    files_rm_parser = files_subparser.add_parser(
        "rm", parents=[shared_parser], help="Remove files in the collection."
    )
    files_rm_parser.add_argument(
        "rm_paths",
        nargs="+",
        default=None,
        help="Files to be removed from the collection.",
    )
    return main_parser


async def parse_cli_args(args: Optional[Sequence[str]] = None):
    main_parser = get_cli_parser()
    main_args = main_parser.parse_args(args)
    if main_args.action is None:
        main_args = main_parser.parse_args(["--help"])

    configs_items: dict[str, Any] = {
        "no_stderr": main_args.no_stderr,
        "action": CliAction(main_args.action),
        "project_root": main_args.project_root,
        "pipe": main_args.pipe,
    }

    match main_args.action:
        case "vectorise":
            configs_items["files"] = main_args.file_paths
            configs_items["recursive"] = main_args.recursive
            configs_items["include_hidden"] = main_args.include_hidden
            configs_items["force"] = main_args.force
            configs_items["chunk_size"] = main_args.chunk_size
            configs_items["overlap_ratio"] = main_args.overlap
            configs_items["encoding"] = main_args.encoding
        case "query":
            configs_items["query"] = main_args.query
            configs_items["n_result"] = main_args.number
            configs_items["query_multiplier"] = main_args.multiplier
            configs_items["query_exclude"] = main_args.exclude
            configs_items["use_absolute_path"] = main_args.absolute
            configs_items["include"] = [QueryInclude(i) for i in main_args.include]
            configs_items["encoding"] = main_args.encoding
        case "check":
            configs_items["check_item"] = main_args.check_item
        case "init":
            configs_items["force"] = main_args.force
        case "chunks":
            configs_items["files"] = main_args.file_paths
            configs_items["chunk_size"] = main_args.chunk_size
            configs_items["overlap_ratio"] = main_args.overlap
            configs_items["encoding"] = main_args.encoding
        case "prompts":
            configs_items["prompt_categories"] = main_args.prompt_categories
        case "files":
            configs_items["files_action"] = main_args.files_action
            match main_args.files_action:
                case FilesAction.rm:
                    configs_items["rm_paths"] = main_args.rm_paths
    return Config(**configs_items)


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


async def load_config_file(path: Optional[Union[str, Path]] = None):
    """Load config file from ~/.config/vectorcode/config.json(5)"""
    if path is None:
        for name in ("config.json5", "config.json"):
            p = os.path.join(GLOBAL_CONFIG_DIR, name)
            if os.path.isfile(p):
                path = str(p)
                break
    if path and os.path.isfile(path):
        logger.debug(f"Loading config from {path}")
        with open(path) as fin:
            content = fin.read()
        if content:
            config = json5.loads(content)
            if isinstance(config, dict):
                expand_envs_in_dict(config)
                return await Config.import_from(config)
            else:
                logger.error("Invalid configuration format!")
                raise ValueError("Invalid configuration format!")
        else:
            logger.debug("Skipping empty json file.")
    else:
        logger.warning("Loading default config.")
    return Config()


async def find_project_config_dir(start_from: Union[str, Path] = "."):
    """Returns the project-local config directory."""
    current_dir = Path(start_from).resolve()
    project_root_anchors = [".vectorcode", ".git"]
    while current_dir:
        for anchor in project_root_anchors:
            to_be_checked = os.path.join(current_dir, anchor)
            if os.path.isdir(to_be_checked):
                logger.debug(f"Found root anchor at {str(to_be_checked)}")
                return to_be_checked
        parent = current_dir.parent
        if parent.resolve() == current_dir:
            logger.debug(
                f"Couldn't find project root after reaching {str(current_dir)}"
            )
            return
        current_dir = parent.resolve()


def find_project_root(
    start_from: Union[str, Path], root_anchor: Union[str, Path] = ".vectorcode"
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


async def get_project_config(project_root: Union[str, Path]) -> Config:
    """
    Load config file for `project_root`.
    Fallback to global config, and then default config.
    """
    if not os.path.isabs(project_root):
        project_root = os.path.abspath(project_root)
    exts = ("json5", "json")
    config = None
    for ext in exts:
        local_config_path = os.path.join(project_root, ".vectorcode", f"config.{ext}")
        if os.path.isfile(local_config_path):
            config = await load_config_file(local_config_path)
            break
    if config is None:
        config = await load_config_file()
    config.project_root = project_root
    return config


def expand_path(path: Union[str, Path], absolute: bool = False) -> Union[str, Path]:
    expanded = os.path.expanduser(os.path.expandvars(path))
    if absolute:
        return os.path.abspath(expanded)
    return expanded


async def expand_globs(
    paths: Sequence[os.PathLike | str],
    recursive: bool = False,
    include_hidden: bool = False,
) -> list[str]:
    result = set()
    stack = list(str(i) for i in paths)
    while stack:
        curr = stack.pop()
        if os.path.isfile(curr):
            result.add(expand_path(curr))
        elif "**" in str(curr):
            stack.extend(glob.glob(curr, recursive=True, include_hidden=include_hidden))
        elif "*" in str(curr):
            stack.extend(
                glob.glob(curr, recursive=recursive, include_hidden=include_hidden)
            )
        elif os.path.isdir(curr) and recursive:
            stack.extend(
                glob.glob(
                    os.path.join(curr, "**", "*"),
                    recursive=recursive,
                    include_hidden=include_hidden,
                )
            )
    return list(result)


def cleanup_path(path: str):
    if os.path.isabs(path) and os.environ.get("HOME") is not None:
        return path.replace(os.environ["HOME"], "~")
    return path


def config_logging(
    name: str = "vectorcode", stdio: bool = True, log_file: bool = False
):  # pragma: nocover
    """Configure the logging module. This should be called before a `main` function (CLI, LSP or MCP)."""

    logging.root.handlers = []

    level_from_env = os.environ.get("VECTORCODE_LOG_LEVEL")
    level = None
    if level_from_env:
        level = logging._nameToLevel.get(level_from_env.upper())
        if level is None:
            logging.warning(
                "Invalid log level: %s. Falling back to default levels.", level_from_env
            )

    handlers = []
    if level is not None or log_file:
        log_dir = os.path.expanduser("~/.local/share/vectorcode/logs/")
        os.makedirs(log_dir, exist_ok=True)
        # File handler
        log_file_path = os.path.join(
            log_dir, f"{name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level or logging.WARN)
        handlers.append(file_handler)
        if stdio:
            atexit.register(lambda: print(f"Saving log to {log_file_path}."))

    if stdio:
        import colorlog

        # Console handler
        console_handler = colorlog.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                fmt="%(log_color)s%(levelname)s%(reset)s: %(name)s : %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
                reset=True,
            )
        )
        console_handler.setLevel(level or logging.WARN)
        handlers.append(console_handler)

    logging.basicConfig(
        handlers=handlers,
        level=level,
    )


class LockManager:
    """
    A class that manages file locks that protects the database files in daemon processes (LSP, MCP).
    """

    __locks: dict[str, AsyncFileLock]
    singleton: Optional["LockManager"] = None

    def __new__(cls) -> "LockManager":
        if cls.singleton is None:
            cls.singleton = super().__new__(cls)
            cls.singleton.__locks = {}
        return cls.singleton

    def get_lock(self, path: str | os.PathLike) -> AsyncFileLock:
        path = str(expand_path(str(path), True))
        if os.path.isdir(path):
            lock_file = os.path.join(path, "vectorcode.lock")
            logger.info(f"Creating {lock_file} for locking.")
            if not os.path.isfile(lock_file):
                with open(lock_file, mode="w") as fin:
                    fin.write("")
            path = lock_file
        if self.__locks.get(path) is None:
            self.__locks[path] = AsyncFileLock(path)  # pyright: ignore[reportArgumentType]
        return self.__locks[path]


class SpecResolver:
    """
    This class is a wrapper around filespec that makes it easier to work with file specs that are not in cwd.
    """

    @classmethod
    def from_path(cls, spec_path: str, project_root: Optional[str] = None):
        """
        Automatically determine the appropriate `base_dir` for resolving file specs that are outside of the project root.
        Only supports `.gitignore` and `.vectorcode/vectorcode.{include,exclude}`.
        Raises `ValueError` if the spec path is not one of them.
        """
        base_dir = "."
        if spec_path.endswith(".gitignore"):
            base_dir = spec_path.replace(".gitignore", "")
        else:
            path_obj = Path(spec_path)
            if path_obj.name in {"vectorcode.include", "vectorcode.exclude"}:
                if path_obj.parent.name == ".vectorcode":
                    # project config
                    base_dir = str(path_obj.parent.parent)
                else:
                    # assume to be global config
                    base_dir = project_root or "."
            else:  # pragma: nocover
                raise ValueError(f"Unsupported spec path: {spec_path}")
        return cls(spec_path, base_dir)

    def __init__(self, spec: str | GitIgnoreSpec, base_dir: str = "."):
        if isinstance(spec, str):
            with open(spec) as fin:
                self.spec = GitIgnoreSpec.from_lines(
                    (i.strip() for i in fin.readlines())
                )
        else:
            self.spec = spec
        self.base_dir = base_dir

    def match(
        self, paths: Iterable[str], negated: bool = False
    ) -> Generator[str, None, None]:
        # get paths relative to `base_dir`

        base = Path(self.base_dir).resolve()
        for p in paths:
            if base in Path(p).resolve().parents:
                should_yield = self.spec.match_file(os.path.relpath(p, self.base_dir))
                if negated:
                    should_yield = not should_yield
                if should_yield:
                    yield p
            else:
                yield p
