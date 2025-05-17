import os
import tempfile
from unittest.mock import MagicMock

import pytest
from tree_sitter import Point

from vectorcode.chunking import (
    Chunk,
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
        assert list(string_chunker.chunk("hello world"))[0] == Chunk(
            "hello", Point(1, 0), Point(1, 4)
        )

        string_chunker = StringChunker(Config(chunk_size=5, overlap_ratio=0))
        assert list(str(i) for i in string_chunker.chunk("hello world")) == [
            "hello",
            " worl",
            "d",
        ]
        chunks = list(string_chunker.chunk("hello world"))
        assert chunks[1] == Chunk(" worl", Point(1, 5), Point(1, 9))

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
        test_content = ["hello ", "world"]

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_file.writelines(test_content)
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

    def test_file_chunker_positions(self):
        test_content = ["first line\n", "second line\n", "third line"]

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_file.writelines(test_content)
            tmp_file_name = tmp_file.name

        # Test chunk positions
        with open(tmp_file_name, "r") as f:
            chunker = FileChunker(Config(chunk_size=10, overlap_ratio=0))
            chunks = list(chunker.chunk(f))

            assert chunks[0].text == "first line"
            assert chunks[0].start == Point(1, 0)
            assert chunks[0].end == Point(1, 9)

            assert chunks[1].text == "\nsecond li"
            assert chunks[1].start == Point(1, 10)
            assert chunks[1].end == Point(2, 8)

            assert chunks[2].text == "ne\nthird l"
            assert chunks[2].start == Point(2, 9)
            assert chunks[2].end == Point(3, 6)

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


def test_treesitter_chunker_fallback_on_long_node():
    test_content = r"""
def foo():
    return "a very very very very very long string"
    """
    config = Config(chunk_size=15)
    with (
        tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        ) as temp_py_file,
    ):
        temp_py_file.write(test_content)
    ts_chunker = TreeSitterChunker(config)
    ts_chunker._fallback_chunker.chunk = MagicMock()
    list(ts_chunker.chunk(temp_py_file.name))
    ts_chunker._fallback_chunker.chunk.assert_called_once()


def test_treesitter_chunker_python_encoding():
    """Test TreeSitterChunker with a sample file using tempfile."""
    chunker = TreeSitterChunker(Config(chunk_size=30, encoding="gbk"))

    test_content = r"""
def 测试():
    return "foo"

def bar():
    return "bar"
    """

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".py", encoding="gbk"
    ) as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['def 测试():\n    return "foo"', 'def bar():\n    return "bar"']
    os.remove(test_file)


def test_treesitter_chunker_python_auto_encoding():
    """Test TreeSitterChunker with a sample file using tempfile."""
    chunker = TreeSitterChunker(Config(chunk_size=30, encoding="_auto"))

    test_content = r"""
def 测试():
    return "foo"

def bar():
    return "bar"
    """

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".py", encoding="utf-16"
    ) as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['def 测试():\n    return "foo"', 'def bar():\n    return "bar"']
    os.remove(test_file)

def test_treesitter_chunker_javascript():
    """Test TreeSitterChunker with a sample javascript file using tempfile."""
    chunker = TreeSitterChunker(Config(chunk_size=60))

    test_content = r"""
function foo() {
    return "foo";
}

function bar() {
    return "bar";
}
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".js") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['function foo() {\n    return "foo";\n}', 'function bar() {\n    return "bar";\n}']
    os.remove(test_file)

def test_treesitter_chunker_javascript_genshi():
    """Test TreeSitterChunker with a sample javascript + genshi file using tempfile. (bypassing lexers via the filetype_map config param)"""
    chunker = TreeSitterChunker(Config(chunk_size=60, filetype_map={"javascript": ["^kid$"]}))

    test_content = r"""
function foo() {
    return `foo with ${genshi}`;
}

function bar() {
    return "bar";
}
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".kid") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['function foo() {\n    return `foo with ${genshi}`;\n}', 'function bar() {\n    return "bar";\n}']
    os.remove(test_file)

def test_treesitter_chunker_parser_from_config_no_parser_found_error():
    """Test TreeSitterChunker filetype_map: should raise an error if no parser is found"""
    chunker = TreeSitterChunker(Config(chunk_size=60, filetype_map={"unknown_parser": ["^kid$"]}))

    test_content = r"""
function foo() {
    return `foo with ${genshi}`;
}

function bar() {
    return "bar";
}
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".kid") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name


    with pytest.raises(ValueError):
        chunks = list(str(i) for i in chunker.chunk(test_file))
        assert chunks == []
    os.remove(test_file)

def test_treesitter_chunker_parser_from_config_regex_error():
    """Test TreeSitterChunker filetype_map: should raise an error if a regex is invalid"""
    chunker = TreeSitterChunker(Config(chunk_size=60, filetype_map={"javascript": ["\\"]}))

    test_content = r"""
function foo() {
    return `foo with ${genshi}`;
}

function bar() {
    return "bar";
}
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".kid") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name


    with pytest.raises(ValueError):
        chunks = list(str(i) for i in chunker.chunk(test_file))
        assert chunks == []
    os.remove(test_file)

def test_treesitter_chunker_parser_from_config_no_language_match():
    """Test TreeSitterChunker filetype_map: should continue with the lexer parser checks if no language matches a regex"""
    chunker = TreeSitterChunker(Config(chunk_size=60, filetype_map={"php": ["^jsx$"]}))

    test_content = r"""
function foo() {
    return "foo";
}

function bar() {
    return "bar";
}
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".js") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(str(i) for i in chunker.chunk(test_file))
    assert chunks == ['function foo() {\n    return "foo";\n}', 'function bar() {\n    return "bar";\n}']
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
    chunker = TreeSitterChunker(Config(chunk_size=35, chunk_filters={"*": [".*foo.*"]}))

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
    assert chunks == ['function bar()\n  return "bar"\nend']
    os.remove(test_file)


def test_treesitter_chunker_lua():
    chunker = TreeSitterChunker(Config(chunk_size=35))
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
    assert chunks == [
        'function foo()\n  return "foo"\nend',
        'function bar()\n  return "bar"\nend',
    ]

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


def test_treesitter_chunker_positions():
    """Test that TreeSitterChunker produces correct start/end positions for chunks."""
    chunker = TreeSitterChunker(Config(chunk_size=15))

    test_content = """\
def foo():
    return 1 + \\
        2

@decorator
def bar():
    return "bar"
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp_file:
        tmp_file.write(test_content)
        test_file = tmp_file.name

    chunks = list(chunker.chunk(test_file))

    # Verify chunks and their positions
    assert len(chunks) >= 2  # Should have at least 2 chunks

    # First chunk should contain the function definition start
    assert "def foo():" in chunks[0].text
    assert chunks[0].start == Point(1, 0)

    # Last chunk should contain the final return statement
    assert 'return "bar"' in chunks[-1].text
    assert chunks[-1].end.row == 7
    assert chunks[-1].end.column in (14, 15)  # Allow 1-column difference

    # Verify positions are contiguous
    for i in range(len(chunks) - 1):
        assert chunks[i].end.row <= chunks[i + 1].start.row
        if chunks[i].end.row == chunks[i + 1].start.row:
            assert chunks[i].end.column <= chunks[i + 1].start.column

    os.remove(test_file)
