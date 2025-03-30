import os
import re
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from io import TextIOWrapper
from typing import Generator, Optional

from pygments.lexer import Lexer
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound
from tree_sitter import Node, Point
from tree_sitter_language_pack import get_parser

from vectorcode.cli_utils import Config


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


class ChunkerBase:
    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = Config()
        assert 0 <= config.overlap_ratio < 1, (
            "Overlap ratio has to be a float between 0 (inclusive) and 1 (exclusive)."
        )
        self.config = config

    @abstractmethod
    def chunk(self, data) -> Generator[Chunk, None, None]:
        raise NotImplementedError


class StringChunker(ChunkerBase):
    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = Config()
        super().__init__(config)

    def chunk(self, data: str):
        if self.config.chunk_size < 0:
            yield Chunk(
                text=data,
                start=Point(row=1, column=0),
                end=Point(row=1, column=len(data)),
            )
        else:
            step_size = max(
                1, int(self.config.chunk_size * (1 - self.config.overlap_ratio))
            )
            i = 0
            while i < len(data):
                chunk_text = data[i : i + self.config.chunk_size]
                yield Chunk(
                    text=chunk_text,
                    start=Point(row=1, column=i),
                    end=Point(row=1, column=i + len(chunk_text) - 1),
                )
                if i + self.config.chunk_size >= len(data):
                    break
                i += step_size


class FileChunker(ChunkerBase):
    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = Config()
        super().__init__(config)

    def chunk(self, data: TextIOWrapper) -> Generator[Chunk, None, None]:
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

    def __chunk_node(self, node: Node, text: str) -> Generator[Chunk, None, None]:
        current_chunk = ""

        current_start = None

        for child in node.children:
            child_text = text[child.start_byte : child.end_byte]
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
                yield from self.__chunk_node(child, text)

            elif not current_chunk:
                # Start new chunk
                current_chunk = child_text
                current_start = Point(
                    row=child.start_point.row + 1, column=child.start_point.column
                )

            elif len(current_chunk) + child_length <= self.config.chunk_size:
                # Add to current chunk
                current_chunk += child_text

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
                current_chunk = child_text
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
            return guess_lexer_for_filename(path, content)

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
            patterns = [f"(?:{i})" for i in patterns]
            return f"(?:{'|'.join(patterns)})"
        return ""

    def chunk(self, data: str) -> Generator[Chunk, None, None]:
        """
        data: path to the file
        """
        assert os.path.isfile(data)
        with open(data) as fin:
            lines = fin.readlines()
            content = "".join(lines)
            if self.config.chunk_size < 0 and content:
                yield Chunk(content, Point(1, 0), Point(len(lines), len(lines[-1]) - 1))
                return
        parser = None
        language = None
        lexer = self.__guess_type(data, content)
        if lexer is not None:
            lang_names = [lexer.name]
            lang_names.extend(lexer.aliases)
            for name in lang_names:
                try:
                    parser = get_parser(name.lower())
                    if parser is not None:
                        language = name.lower()
                        break
                except LookupError:  # pragma: nocover
                    pass

        if parser is None:
            # fall back to naive chunking
            yield from StringChunker(self.config).chunk(content)
        else:
            pattern_str = self.__build_pattern(language=language)
            content_bytes = content.encode()
            tree = parser.parse(content_bytes)
            chunks_gen = self.__chunk_node(tree.root_node, content)
            if pattern_str:
                re_pattern = re.compile(pattern_str)
                for chunk in chunks_gen:
                    if re_pattern.match(chunk.text) is None:
                        yield chunk
            else:
                yield from chunks_gen
