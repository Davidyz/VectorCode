from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException, InvalidDimensionException

from vectorcode.cli_utils import Config, QueryInclude
from vectorcode.subcommands.query import get_query_result_files, query


@pytest.fixture
def mock_collection():
    collection = AsyncMock(spec=AsyncCollection)
    collection.count.return_value = 10
    collection.query.return_value = {
        "ids": [["id1", "id2", "id3"], ["id4", "id5", "id6"]],
        "distances": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "metadatas": [
            [{"path": "file1.py"}, {"path": "file2.py"}, {"path": "file3.py"}],
            [{"path": "file2.py"}, {"path": "file4.py"}, {"path": "file3.py"}],
        ],
        "documents": [
            ["content1", "content2", "content3"],
            ["content4", "content5", "content6"],
        ],
    }
    return collection


@pytest.fixture
def mock_config():
    return Config(
        query=["test query"],
        n_result=3,
        query_multiplier=2,
        chunk_size=100,
        overlap_ratio=0.2,
        project_root="/test/project",
        pipe=False,
        include=[QueryInclude.path, QueryInclude.document],
        query_exclude=[],
        reranker=None,
        reranker_params={},
        use_absolute_path=False,
    )


@pytest.mark.asyncio
async def test_get_query_result_files(mock_collection, mock_config):
    # Mock the reranker
    with patch("vectorcode.subcommands.query.reranker.NaiveReranker") as MockReranker:
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [
            "file1.py",
            "file2.py",
            "file3.py",
        ]
        MockReranker.return_value = mock_reranker_instance

        # Call the function
        result = await get_query_result_files(mock_collection, mock_config)

        # Check that query was called with the right parameters
        mock_collection.query.assert_called_once()
        args, kwargs = mock_collection.query.call_args
        assert kwargs["query_texts"] == [
            "test query"
        ]  # Assuming chunking produces this
        assert kwargs["n_results"] == 6  # n_result(3) * query_multiplier(2)
        assert IncludeEnum.metadatas in kwargs["include"]
        assert IncludeEnum.distances in kwargs["include"]
        assert IncludeEnum.documents in kwargs["include"]
        assert kwargs["where"] is None  # Since query_exclude is empty

        # Check reranker was used correctly
        MockReranker.assert_called_once_with(mock_config)
        mock_reranker_instance.rerank.assert_called_once_with(
            mock_collection.query.return_value
        )

        # Check the result
        assert result == ["file1.py", "file2.py", "file3.py"]


@pytest.mark.asyncio
async def test_get_query_result_files_with_query_exclude(mock_collection, mock_config):
    # Setup query_exclude
    mock_config.query_exclude = ["/excluded/path.py"]

    with (
        patch("vectorcode.subcommands.query.expand_path") as mock_expand_path,
        patch("vectorcode.subcommands.query.expand_globs") as mock_expand_globs,
        patch("vectorcode.subcommands.query.reranker.NaiveReranker") as MockReranker,
        patch("os.path.isfile", return_value=True),  # Add this line to mock isfile
    ):
        mock_expand_globs.return_value = ["/excluded/path.py"]
        mock_expand_path.return_value = "/excluded/path.py"

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = ["file1.py", "file2.py"]
        MockReranker.return_value = mock_reranker_instance

        # Call the function
        await get_query_result_files(mock_collection, mock_config)

        # Check that query was called with the right parameters including the where clause
        mock_collection.query.assert_called_once()
        _, kwargs = mock_collection.query.call_args
        assert kwargs["where"] == {"path": {"$nin": ["/excluded/path.py"]}}


@pytest.mark.asyncio
async def test_get_query_result_files_with_cross_encoder_reranker(
    mock_collection, mock_config
):
    # Configure to use CrossEncoder reranker
    mock_config.reranker = "cross-encoder/model-name"

    with patch(
        "vectorcode.subcommands.query.reranker.CrossEncoderReranker"
    ) as MockCrossEncoder:
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = ["file1.py", "file2.py"]
        MockCrossEncoder.return_value = mock_reranker_instance

        # Call the function
        result = await get_query_result_files(mock_collection, mock_config)

        # Check CrossEncoder was used correctly
        MockCrossEncoder.assert_called_once()
        # Check the result
        assert result == ["file1.py", "file2.py"]


@pytest.mark.asyncio
async def test_get_query_result_files_empty_collection(mock_collection, mock_config):
    # Setup an empty collection
    mock_collection.count.return_value = 0

    # Call the function
    result = await get_query_result_files(mock_collection, mock_config)

    # Check that the result is an empty list
    assert result == []
    # Ensure query wasn't called
    mock_collection.query.assert_not_called()


@pytest.mark.asyncio
async def test_get_query_result_files_query_error(mock_collection, mock_config):
    # Make query raise an IndexError
    mock_collection.query.side_effect = IndexError("No results")

    # Call the function
    result = await get_query_result_files(mock_collection, mock_config)

    # Check that the result is an empty list
    assert result == []


@pytest.mark.asyncio
async def test_get_query_result_files_chunking(mock_collection, mock_config):
    # Set a long query that will be chunked
    mock_config.query = [
        "this is a longer query that should be chunked into multiple parts"
    ]

    with (
        patch("vectorcode.subcommands.query.StringChunker") as MockChunker,
        patch("vectorcode.subcommands.query.reranker.NaiveReranker") as MockReranker,
    ):
        # Set up MockChunker to chunk the query
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.return_value = ["chunk1", "chunk2", "chunk3"]
        MockChunker.return_value = mock_chunker_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = ["file1.py", "file2.py"]
        MockReranker.return_value = mock_reranker_instance

        # Call the function
        result = await get_query_result_files(mock_collection, mock_config)

        # Check that the chunker was used correctly
        MockChunker.assert_called_once_with(mock_config)
        mock_chunker_instance.chunk.assert_called_once_with(mock_config.query[0])

        # Check query was called with chunked query
        mock_collection.query.assert_called_once()
        _, kwargs = mock_collection.query.call_args
        assert kwargs["query_texts"] == ["chunk1", "chunk2", "chunk3"]

        # Check the result
        assert result == ["file1.py", "file2.py"]


@pytest.mark.asyncio
async def test_get_query_result_files_multiple_queries(mock_collection, mock_config):
    # Set multiple query terms
    mock_config.query = ["term1", "term2", "term3"]

    with (
        patch("vectorcode.subcommands.query.StringChunker") as MockChunker,
        patch("vectorcode.subcommands.query.reranker.NaiveReranker") as MockReranker,
    ):
        # Set up MockChunker to return the query terms as is
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.side_effect = lambda q: [q]
        MockChunker.return_value = mock_chunker_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = ["file1.py", "file2.py"]
        MockReranker.return_value = mock_reranker_instance

        # Call the function
        result = await get_query_result_files(mock_collection, mock_config)

        # Check that chunker was called for each query term
        assert mock_chunker_instance.chunk.call_count == 3

        # Check query was called with all query terms
        mock_collection.query.assert_called_once()
        _, kwargs = mock_collection.query.call_args
        assert set(kwargs["query_texts"]) == set(["term1", "term2", "term3"])

        # Check the result
        assert result == ["file1.py", "file2.py"]


@pytest.mark.asyncio
async def test_query_success(mock_config):
    # Mock all the necessary dependencies
    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.query.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.query.get_collection", return_value=mock_collection
        ),
        patch("vectorcode.subcommands.query.verify_ef", return_value=True),
        patch("vectorcode.subcommands.query.get_query_result_files") as mock_get_files,
        patch("builtins.open", create=True) as mock_open,
        patch("json.dumps"),
        patch("os.path.isfile", return_value=True),
        patch("os.path.relpath", return_value="rel/path.py"),
        patch("os.path.abspath", return_value="/abs/path.py"),
    ):
        # Set up the mock file paths and contents
        mock_get_files.return_value = ["file1.py", "file2.py"]
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = "file content"
        mock_open.return_value = mock_file_handle

        # Call the function
        result = await query(mock_config)

        # Verify the function completed successfully
        assert result == 0

        # Check that all the expected functions were called
        mock_get_files.assert_called_once_with(mock_collection, mock_config)

        # Check file opening and reading
        assert mock_open.call_count == 2  # Two files


@pytest.mark.asyncio
async def test_query_pipe_mode(mock_config):
    # Set pipe mode to True
    mock_config.pipe = True

    # Similar to test_query_success but check for JSON output
    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.query.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.query.get_collection", return_value=mock_collection
        ),
        patch("vectorcode.subcommands.query.verify_ef", return_value=True),
        patch("vectorcode.subcommands.query.get_query_result_files") as mock_get_files,
        patch("builtins.open", create=True) as mock_open,
        patch("json.dumps") as mock_json_dumps,
        patch("os.path.isfile", return_value=True),
        patch("os.path.relpath", return_value="rel/path.py"),
        patch("os.path.abspath", return_value="/abs/path.py"),
    ):
        # Set up the mock file paths and contents
        mock_get_files.return_value = ["file1.py", "file2.py"]
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = "file content"
        mock_open.return_value = mock_file_handle

        # Call the function
        result = await query(mock_config)

        # Verify the function completed successfully
        assert result == 0

        # Check that JSON dumps was called
        mock_json_dumps.assert_called_once()


@pytest.mark.asyncio
async def test_query_absolute_path(mock_config):
    # Set use_absolute_path to True
    mock_config.use_absolute_path = True

    # Mock all the necessary dependencies
    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.query.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.query.get_collection", return_value=mock_collection
        ),
        patch("vectorcode.subcommands.query.verify_ef", return_value=True),
        patch("vectorcode.subcommands.query.get_query_result_files") as mock_get_files,
        patch("builtins.open", create=True) as mock_open,
        patch("os.path.isfile", return_value=True),
        patch("os.path.relpath", return_value="rel/path.py"),
        patch("os.path.abspath", return_value="/abs/path.py"),
    ):
        # Set up the mock file paths and contents
        mock_get_files.return_value = ["file1.py"]
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = "file content"
        mock_open.return_value = mock_file_handle

        # Call the function
        result = await query(mock_config)

        # Verify the function completed successfully
        assert result == 0


@pytest.mark.asyncio
async def test_query_collection_not_found():
    config = Config(project_root="/test/project")

    with (
        patch("vectorcode.subcommands.query.get_client"),
        patch("vectorcode.subcommands.query.get_collection") as mock_get_collection,
        patch("sys.stderr"),
    ):
        # Make get_collection raise ValueError
        mock_get_collection.side_effect = ValueError("Collection not found")

        # Call the function
        result = await query(config)

        # Check the error was handled properly
        assert result == 1


@pytest.mark.asyncio
async def test_query_invalid_collection():
    config = Config(project_root="/test/project")

    with (
        patch("vectorcode.subcommands.query.get_client"),
        patch("vectorcode.subcommands.query.get_collection") as mock_get_collection,
        patch("sys.stderr"),
    ):
        # Make get_collection raise InvalidCollectionException
        mock_get_collection.side_effect = InvalidCollectionException(
            "Invalid collection"
        )

        # Call the function
        result = await query(config)

        # Check the error was handled properly
        assert result == 1


@pytest.mark.asyncio
async def test_query_invalid_dimension():
    config = Config(project_root="/test/project")

    with (
        patch("vectorcode.subcommands.query.get_client"),
        patch("vectorcode.subcommands.query.get_collection") as mock_get_collection,
        patch("sys.stderr"),
    ):
        # Make get_collection raise InvalidDimensionException
        mock_get_collection.side_effect = InvalidDimensionException("Invalid dimension")

        # Call the function
        result = await query(config)

        # Check the error was handled properly
        assert result == 1


@pytest.mark.asyncio
async def test_query_invalid_file(mock_config):
    # Set up mocks for a successful query but with an invalid file
    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.query.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.query.get_collection", return_value=mock_collection
        ),
        patch("vectorcode.subcommands.query.verify_ef", return_value=True),
        patch("vectorcode.subcommands.query.get_query_result_files") as mock_get_files,
        patch("os.path.isfile", return_value=False),
        patch("sys.stderr") as mock_stderr,
    ):
        # Set up the mock file paths
        mock_get_files.return_value = ["invalid_file.py"]

        # Call the function
        result = await query(mock_config)

        # Verify the function completed successfully despite invalid file
        assert result == 0

        # Check that a warning was printed to stderr
        assert mock_stderr.write.called


@pytest.mark.asyncio
async def test_query_invalid_ef(mock_config):
    # Test when verify_ef returns False
    mock_client = AsyncMock()
    mock_collection = AsyncMock()

    with (
        patch("vectorcode.subcommands.query.get_client", return_value=mock_client),
        patch(
            "vectorcode.subcommands.query.get_collection", return_value=mock_collection
        ),
        patch("vectorcode.subcommands.query.verify_ef", return_value=False),
    ):
        # Call the function
        result = await query(mock_config)

        # Verify the function returns error code
        assert result == 1
