import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from io import TextIOWrapper
from typing import Generator, Optional, cast

from pygments.lexer import Lexer
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
from tree_sitter import Node, Point
from tree_sitter_language_pack import SupportedLanguage, get_parser

from vectorcode.cli_utils import Config

logger = logging.getLogger(name=__name__)


@dataclass
class Chunk:
    """
    rows are 1-indexed, cols are 0-indexed.
    """

    text: str
    start: Point
    end: Point

    def __str__(self):
        return self.text

    def export_dict(self):
        if self.start is not None:
            return {
                "text": self.text,
                "start": {"row": self.start.row, "column": self.start.column},
                "end": {"row": self.end.row, "column": self.end.column},
            }
        else:
            return {"text": self.text}


@dataclass
class ChunkOpts:
    start_pos: Point


class ChunkerBase(ABC):  # pragma: nocover
    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = Config()
        assert 0 <= config.overlap_ratio < 1, (
            "Overlap ratio has to be a float between 0 (inclusive) and 1 (exclusive)."
        )
        self.config = config

    @abstractmethod
    def chunk(
        self, data, opts: Optional[ChunkOpts] = None
    ) -> Generator[Chunk, None, None]:
        raise NotImplementedError


class StringChunker(ChunkerBase):
    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = Config()
        super().__init__(config)

    def chunk(self, data: str, opts: Optional[ChunkOpts] = None):
        start_pos = Point(row=1, column=0)
        if opts is not None:
            start_pos = opts.start_pos

        logger.info("Started chunking with StringChunker.")
        logger.debug(f"{data=}")
        if self.config.chunk_size < 0:
            yield Chunk(
                text=data,
                start=start_pos,
                end=Point(
                    row=data.count("\n") + start_pos.row,
                    column=len(data.split("\n")[-1]) - 1,
                ),
            )
        else:
            step_size = max(
                1, int(self.config.chunk_size * (1 - self.config.overlap_ratio))
            )
            i = 0
            while i < len(data):
                chunk_text = data[i : i + self.config.chunk_size]

                start_lines_before_chunk = data[:i].count("\n")
                chunk_start_row = start_pos.row + start_lines_before_chunk
                if start_lines_before_chunk == 0:
                    chunk_start_column = start_pos.column + i
                else:
                    last_newline_idx_before_i = data.rfind("\n", 0, i)
                    chunk_start_column = i - (last_newline_idx_before_i + 1)

                chunk_end_row = chunk_start_row + chunk_text.count("\n")

                if "\n" in chunk_text:
                    chunk_end_column = len(chunk_text.split("\n")[-1]) - 1
                else:
                    chunk_end_column = chunk_start_column + len(chunk_text) - 1

                yield Chunk(
                    text=chunk_text,
                    start=Point(row=chunk_start_row, column=chunk_start_column),
                    end=Point(row=chunk_end_row, column=chunk_end_column),
                )
                if i + self.config.chunk_size >= len(data):
                    break
                i += step_size


class FileChunker(ChunkerBase):
    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = Config()
        super().__init__(config)

    def chunk(
        self, data: TextIOWrapper, opts: Optional[ChunkOpts] = None
    ) -> Generator[Chunk, None, None]:
        logger.info("Started chunking %s using FileChunker.", data.name)
        lines = data.readlines()
        if len(lines) == 0:
            return
        if (
            self.config.chunk_size < 0
            or sum(len(i) for i in lines) < self.config.chunk_size
        ):
            text = "".join(lines)
            yield Chunk(text, Point(1, 0), Point(1, len(text) - 1))
            return

        text = "".join(lines)
        step_size = max(
            1, int(self.config.chunk_size * (1 - self.config.overlap_ratio))
        )

        # Convert lines to absolute positions
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        i = 0
        while i < len(text):
            chunk_text = text[i : i + self.config.chunk_size]

            # Find start position
            start_line = (
                next(ln for ln, offset in enumerate(line_offsets) if offset > i) - 1
            )
            start_col = i - line_offsets[start_line]

            # Find end position
            end_pos = i + len(chunk_text)
            end_line = (
                next(ln for ln, offset in enumerate(line_offsets) if offset >= end_pos)
                - 1
            )
            end_col = end_pos - line_offsets[end_line] - 1

            yield Chunk(
                chunk_text,
                Point(start_line + 1, start_col),
                Point(end_line + 1, end_col),
            )

            if i + self.config.chunk_size >= len(text):
                break
            i += step_size


class TreeSitterChunker(ChunkerBase):
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = Config()
        super().__init__(config)
        self._fallback_chunker = StringChunker(config)

    def __chunk_node(
        self, node: Node, text_bytes: bytes
    ) -> Generator[Chunk, None, None]:
        if node.text is not None:
            logger.debug(
                f"Traversing at node {node.text.decode()} at position {node.byte_range}"
            )
        current_chunk: str = ""
        prev_node = None
        current_start = None

        logger.debug("nbr children: %s", len(node.children))
        # if node has no children we fallback to the string chunker
        if len(node.children) == 0 and node.text:
            logger.debug("No children, falling back to string chunker")
            yield from self._fallback_chunker.chunk(
                node.text.decode(), ChunkOpts(start_pos=node.start_point)
            )

        for child in node.children:
            child_bytes = text_bytes[child.start_byte : child.end_byte]
            child_text = child_bytes.decode()
            child_length = len(child_text)

            if child_length > self.config.chunk_size:
                # Yield current chunk if exists
                if current_chunk:
                    assert current_start is not None
                    yield Chunk(
                        text=current_chunk,
                        start=current_start,
                        end=Point(
                            row=current_start.row + current_chunk.count("\n"),
                            column=len(current_chunk.split("\n")[-1]) - 1
                            if "\n" in current_chunk
                            else current_start.column + len(current_chunk) - 1,
                        ),
                    )
                    current_chunk = ""
                    current_start = None

                # Recursively chunk the large child node
                yield from self.__chunk_node(child, text_bytes)

            elif not current_chunk:
                # Start new chunk
                current_chunk = child_bytes.decode()
                current_start = Point(
                    row=child.start_point.row + 1, column=child.start_point.column
                )
                prev_node = child

            elif len(current_chunk) + child_length + 1 <= self.config.chunk_size:
                # Add to current chunk
                if prev_node:
                    if prev_node.end_point.row != child.start_point.row:
                        current_chunk += "\n"
                    else:
                        current_chunk += " " * (
                            child.start_point.column - prev_node.end_point.column
                        )
                current_chunk += child_bytes.decode()
                prev_node = child

            else:
                # Yield current chunk and start new one
                assert current_start is not None
                yield Chunk(
                    text=current_chunk,
                    start=current_start,
                    end=Point(
                        row=current_start.row + current_chunk.count("\n"),
                        column=len(current_chunk.split("\n")[-1]) - 1
                        if "\n" in current_chunk
                        else current_start.column + len(current_chunk) - 1,
                    ),
                )
                current_chunk = child_bytes.decode()
                current_start = Point(
                    row=child.start_point.row + 1, column=child.start_point.column
                )

        # Yield remaining chunk
        if current_chunk:
            assert current_start is not None
            yield Chunk(
                text=current_chunk,
                start=current_start,
                end=Point(
                    row=current_start.row + current_chunk.count("\n"),
                    column=len(current_chunk.split("\n")[-1]) - 1
                    if "\n" in current_chunk
                    else current_start.column + len(current_chunk) - 1,
                ),
            )

    @cache
    def __guess_type(self, path: str, content: str) -> Optional[Lexer]:
        try:
            return get_lexer_for_filename(path, content)

        except ClassNotFound:
            return None

    @cache
    def __build_pattern(self, language: str):
        patterns = []
        lang_specific_pat = self.config.chunk_filters.get(language)
        if lang_specific_pat:
            patterns.extend(lang_specific_pat)
        else:
            patterns.extend(self.config.chunk_filters.get("*", []))
        if len(patterns):
            logger.debug(
                f"Merging {len(patterns)} filter patterns for excluding chunks."
            )
            patterns = [f"(?:{i})" for i in patterns]
            return f"(?:{'|'.join(patterns)})"
        return ""

    def __load_file_lines(self, path: str) -> list[str]:
        assert os.path.isfile(path), f"{path} is not a valid file!"
        logger.info(f"Started chunking {path} with TreeSitterChunker.")
        encoding = self.config.encoding
        if encoding == "_auto":
            from charset_normalizer import from_path

            match = from_path(path).best()
            if match is None:  # pragma: nocover
                raise UnicodeError(f"Failed to detect the encoding for {path}!")
            logger.info(f"Automatically selected {encoding} for decoding {path}.")
            encoding = match.encoding
        else:
            logger.debug(f"Decoding {path} with {encoding=}.")
        with open(path, encoding=encoding) as fin:
            lines = fin.readlines()
        return lines

    def __get_parser_from_config(self, file_path: str):
        """
        Get parser based on filetype_map config.
        """
        filetype_map = self.config.filetype_map
        if not filetype_map:
            logger.debug("filetype_map is empty in config.")
            return None

        filename = os.path.basename(file_path)
        extension = os.path.splitext(file_path)[1]
        if extension.startswith("."):
            extension = extension[1:]
        logger.debug(f"Checking filetype map for extension '{extension}' in {filename}")
        for _language, patterns in filetype_map.items():
            language = _language.lower()
            for pattern in patterns:
                try:
                    if re.search(pattern, extension):
                        logger.debug(
                            f"'{filename}' extension matches pattern '{pattern}' for language '{language}'. Attempting to load parser."
                        )
                        parser = get_parser(cast(SupportedLanguage, language))
                        logger.debug(
                            f"Found parser for language '{language}' from config."
                        )
                        return parser
                except re.error as e:
                    e.add_note(
                        f"\nInvalid regex pattern '{pattern}' for language '{language}' in filetype_map"
                    )
                    raise
                except LookupError as e:
                    e.add_note(
                        f"\nTreeSitter Parser for language '{language}' not found. Please check your filetype_map config."
                    )
                    raise

        logger.debug(f"No matching filetype map entry found for {filename}.")
        return None

    def chunk(
        self, data: str, opts: Optional[ChunkOpts] = None
    ) -> Generator[Chunk, None, None]:
        """
        data: path to the file
        """
        lines = self.__load_file_lines(data)
        content = "".join(lines)
        if self.config.chunk_size < 0 and content:
            logger.info(
                "Skipping chunking %s because document is smaller than chunk_size.",
                data,
            )
            yield Chunk(content, Point(1, 0), Point(len(lines), len(lines[-1]) - 1))
            return
        parser = None
        language = None
        parser = self.__get_parser_from_config(data)
        if parser is None:
            lexer = self.__guess_type(data, content)
            if lexer is not None:
                lang_names = [lexer.name]
                lang_names.extend(lexer.aliases)
                for name in lang_names:
                    try:
                        parser = get_parser(cast(SupportedLanguage, name.lower()))
                        if parser is not None:
                            language = name.lower()
                            logger.debug(
                                "Detected %s filetype for treesitter chunking.",
                                language,
                            )
                            break
                    except LookupError:  # pragma: nocover
                        pass

        if parser is None:
            logger.debug(
                "Unable to pick a suitable parser. Fall back to naive chunking"
            )
            yield from self._fallback_chunker.chunk(content, opts)
        else:
            pattern_str = self.__build_pattern(language=language)
            content_bytes = content.encode()
            tree = parser.parse(content_bytes)
            chunks_gen = self.__chunk_node(tree.root_node, content_bytes)
            if pattern_str:
                re_pattern = re.compile(pattern_str)
                for chunk in chunks_gen:
                    if re_pattern.match(chunk.text) is None:
                        yield chunk
            else:
                yield from chunks_gen
