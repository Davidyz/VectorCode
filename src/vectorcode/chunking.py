import os
from abc import abstractmethod
from functools import cache
from io import TextIOWrapper
from typing import Generator, Optional

from pygments.lexer import Lexer
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound
from tree_sitter import Node
from tree_sitter_language_pack import get_parser


class ChunkerBase:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def chunk(self, data) -> Generator[str, None, None]:
        raise NotImplementedError


class StringChunker(ChunkerBase):
    def __init__(self, chunk_size: int = -1, overlap_ratio: float = 0.2) -> None:
        super().__init__()
        self.__chunk_size = chunk_size
        assert 0 <= overlap_ratio < 1, (
            "Overlap ratio has to be a float between 0 (inclusive) and 1 (exclusive)."
        )
        self.__overlap_ratio = overlap_ratio

    def chunk(self, data: str) -> Generator[str, None, None]:
        if self.__chunk_size < 0:
            yield data
        else:
            step_size = max(1, int(self.__chunk_size * (1 - self.__overlap_ratio)))
            i = 0
            while i < len(data):
                yield data[i : i + self.__chunk_size]
                if i + self.__chunk_size >= len(data):
                    break
                i += step_size


class FileChunker(ChunkerBase):
    def __init__(self, chunk_size: int = -1, overlap_ratio: float = 0.2) -> None:
        super().__init__()
        self.__chunk_size = chunk_size
        assert 0 <= overlap_ratio < 1, (
            "Overlap ratio has to be a float between 0 (inclusive) and 1 (exclusive)."
        )
        self.__overlap_ratio = overlap_ratio

    def chunk(self, data: TextIOWrapper) -> Generator[str, None, None]:
        if self.__chunk_size < 0:
            yield "".join(data.readlines())
        else:
            step_size = max(1, int(self.__chunk_size * (1 - self.__overlap_ratio)))
            # the output of this method should be identical to that of StringChunker.chunk
            output = data.read(self.__chunk_size)
            yield output
            if len(output) < self.__chunk_size:
                return
            while True:
                new_chars = data.read(step_size)
                output = output[step_size:] + new_chars
                yield output
                if len(new_chars) < step_size:
                    return


class TreeSitterChunker(ChunkerBase):
    def __init__(self, chunk_size: int = -1, overlap_ratio: float = 0.2):
        super().__init__()
        assert isinstance(chunk_size, int), "chunk_size parameter must be an integer"
        assert 0 <= overlap_ratio < 1, (
            "Overlap ratio has to be a float between 0 (inclusive) and 1 (exclusive)."
        )
        self.__chunk_size = chunk_size
        self.__overlap_ratio = overlap_ratio

    def __chunk_node(self, node: Node, text: str) -> Generator[str, None, None]:
        current_chunk = ""
        for child in node.children:
            child_length = child.end_byte - child.start_byte
            if child_length > self.__chunk_size:
                if current_chunk:
                    yield current_chunk
                    current_chunk = ""
                yield from self.__chunk_node(child, text)
            elif len(current_chunk) + child_length > self.__chunk_size:
                yield current_chunk
                current_chunk = text[child.start_byte : child.end_byte]
            else:
                current_chunk += text[child.start_byte : child.end_byte]
        if current_chunk:
            yield current_chunk

    @cache
    def __guess_type(self, path: str, content: str) -> Optional[Lexer]:
        try:
            return guess_lexer_for_filename(path, content)

        except ClassNotFound:
            return None

    def chunk(self, data: str) -> Generator[str, None, None]:
        """
        data: path to the file
        """
        assert os.path.isfile(data)
        with open(data) as fin:
            content = fin.read()
        if self.__chunk_size < 0:
            yield content
            return
        parser = None
        lexer = self.__guess_type(data, content)
        if lexer is not None:
            lang_names = [lexer.name]
            lang_names.extend(lexer.aliases)
            for name in lang_names:
                try:
                    parser = get_parser(name.lower())
                    break
                except LookupError:
                    pass

        if parser is None:
            # fall back to naive chunking
            yield from StringChunker(self.__chunk_size, self.__overlap_ratio).chunk(
                content
            )
        else:
            content_bytes = content.encode()
            tree = parser.parse(content_bytes)
            yield from self.__chunk_node(tree.root_node, content)
