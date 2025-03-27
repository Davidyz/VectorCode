import os
import tempfile

import pytest

from vectorcode.chunking import (
    ChunkerBase,
    FileChunker,
    StringChunker,
    TreeSitterChunker,
)
from vectorcode.cli_utils import Config


class TestStringChunker:
    file_chunker = FileChunker()

    def test_string_chunker(self):
        string_chunker = StringChunker(Config(chunk_size=-1, overlap_ratio=0.5))
        assert list(str(i) for i in string_chunker.chunk("hello world")) == [
            "hello world"
        ]
        string_chunker = StringChunker(Config(chunk_size=5, overlap_ratio=0.5))
        assert list(str(i) for i in string_chunker.chunk("hello world")) == [
            "hello",
            "llo w",
            "o wor",
            "world",
        ]
        string_chunker = StringChunker(Config(chunk_size=5, overlap_ratio=0))
        assert list(str(i) for i in string_chunker.chunk("hello world")) == [
            "hello",
            " worl",
            "d",
        ]

        string_chunker = StringChunker(Config(chunk_size=5, overlap_ratio=0.8))
        assert list(str(i) for i in string_chunker.chunk("hello world")) == [
            "hello",
            "ello ",
            "llo w",
            "lo wo",
            "o wor",
            " worl",
            "world",
        ]


class TestFileChunker:
    def test_file_chunker(self):
        test_content = "hello world"

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file_name = tmp_file.name

        # Test negative chunk size (return whole file)
        with open(tmp_file_name, "r") as f:
            chunker = FileChunker(Config(chunk_size=-1, overlap_ratio=0.5))
            assert list(str(i) for i in chunker.chunk(f)) == ["hello world"]

        # Test basic chunking with overlap
        with open(tmp_file_name, "r") as f:
            chunker = FileChunker(Config(chunk_size=5, overlap_ratio=0.5))
            assert list(str(i) for i in chunker.chunk(f)) == [
                "hello",
                "llo w",
                "o wor",
                "world",
            ]

        # Test no overlap
        with open(tmp_file_name, "r") as f:
            chunker = FileChunker(Config(chunk_size=5, overlap_ratio=0))
            assert list(str(i) for i in chunker.chunk(f)) == ["hello", " worl", "d"]

        # Test high overlap ratio
        with open(tmp_file_name, "r") as f:
            chunker = FileChunker(Config(chunk_size=5, overlap_ratio=0.8))
            assert list(str(i) for i in chunker.chunk(f)) == [
                "hello",
                "ello ",
                "llo w",
                "lo wo",
                "o wor",
                " worl",
                "world",
            ]

        os.remove(tmp_file_name)


def test_no_config():
    assert StringChunker().config == Config()
    assert FileChunker().config == Config()
    assert TreeSitterChunker().config == Config()


def test_chunker_base():
    with pytest.raises(AssertionError):
        ChunkerBase(Config(overlap_ratio=-1))
    with pytest.raises(NotImplementedError):
        ChunkerBase().chunk("hello")
    assert ChunkerBase().config == Config()


def test_treesitter_chunker_python():
    """Test TreeSitterChunker with a sample file using tempfile."""
    chunker = TreeSitterChunker(Config(chunk_size=30))

    test_content = r"""
def foo():
    return "foo"

def bar():
    return "bar"
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['def foo():\n    return "foo"', 'def bar():\n    return "bar"']
    os.remove(test_file)


def test_treesitter_chunker_filter():
    chunker = TreeSitterChunker(
        Config(chunk_size=30, chunk_filters={"python": [".*foo.*"]})
    )

    test_content = r"""
def foo():
    return "foo"

def bar():
    return "bar"
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['def bar():\n    return "bar"']
    os.remove(test_file)


def test_treesitter_chunker_filter_merging():
    chunker = TreeSitterChunker(
        Config(chunk_size=30, chunk_filters={"python": [".*foo.*", ".*bar.*"]})
    )

    test_content = r"""
def foo():
    return "foo"

def bar():
    return "bar"
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == []
    os.remove(test_file)


def test_treesitter_chunker_filter_wildcard():
    chunker = TreeSitterChunker(Config(chunk_size=30, chunk_filters={"*": [".*foo.*"]}))

    test_content = r"""
def foo():
    return "foo"

def bar():
    return "bar"
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['def bar():\n    return "bar"']
    os.remove(test_file)

    test_content = r"""
function foo()
  return "foo"
end

function bar()
  return "bar"
end
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".lua") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['functionbar()return "bar"end']
    os.remove(test_file)


def test_treesitter_chunker_lua():
    chunker = TreeSitterChunker(Config(chunk_size=30))
    test_content = r"""
function foo()
  return "foo"
end

function bar()
  return "bar"
end
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".lua") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['functionfoo()return "foo"end', 'functionbar()return "bar"end']

    os.remove(test_file)


def test_treesitter_chunker_ruby():
    chunker = TreeSitterChunker(Config(chunk_size=30))
    test_content = r"""
def greet_user(name)
  "Hello, #{name.capitalize}!"
end

def add_numbers(a, b)
  a + b
end
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".rb") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert len(chunks) > 0

    os.remove(test_file)


def test_treesitter_chunker_neg_chunksize():
    chunker = TreeSitterChunker(Config(chunk_size=-1))
    test_content = r"""
def greet_user(name)
  "Hello, #{name.capitalize}!"
end

def add_numbers(a, b)
  a + b
end
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".rb") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert len(chunks) == 1

    os.remove(test_file)


def test_treesitter_chunker_fallback():
    """Test that TreeSitterChunker falls back to StringChunker when no parser is found."""
    chunk_size = 30
    overlap_ratio = 0.2
    tree_sitter_chunker = TreeSitterChunker(
        Config(chunk_size=chunk_size, overlap_ratio=overlap_ratio)
    )
    string_chunker = StringChunker(
        Config(chunk_size=chunk_size, overlap_ratio=overlap_ratio)
    )

    test_content = "This is a test string."

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".xyz"
    ) as tmp_file:  # Use an uncommon extension
        tmp_file.write(test_content)
        test_file = tmp_file.name

    tree_sitter_chunks = list(str(i) for i in tree_sitter_chunker.chunk(test_file))
    string_chunks = list(str(i) for i in string_chunker.chunk(test_content))

    assert tree_sitter_chunks == string_chunks

    os.remove(test_file)
