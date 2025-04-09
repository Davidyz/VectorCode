"""Tests for error handling in the query module's reranker integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from chromadb.api.models.AsyncCollection import AsyncCollection

from vectorcode.cli_utils import Config, QueryInclude
from vectorcode.subcommands.query import get_query_result_files


@pytest.fixture
def mock_collection():
    collection = AsyncMock(spec=AsyncCollection)
    collection.count.return_value = 10
    collection.query.return_value = {
        "ids": [["id1", "id2", "id3"]],
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [
            [
                {"path": "file1.py", "start": 1, "end": 1},
                {"path": "file2.py", "start": 1, "end": 1},
                {"path": "file3.py", "start": 1, "end": 1},
            ],
        ],
        "documents": [
            ["content1", "content2", "content3"],
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
async def test_get_query_result_files_registry_error(mock_collection, mock_config):
    """Test graceful handling of a reranker not found in registry."""
    # Set a custom reranker to trigger the error path
    mock_config.reranker = "custom-reranker"

    # Mock stderr to capture error messages
    with patch("sys.stderr") as mock_stderr:
        # Mock the NaiveReranker for fallback
        with patch("vectorcode.subcommands.query.reranker.NaiveReranker") as mock_naive:
            mock_reranker_instance = MagicMock()
            mock_reranker_instance.rerank.return_value = ["file1.py", "file2.py"]
            mock_naive.return_value = mock_reranker_instance

            # This should fall back to NaiveReranker
            result = await get_query_result_files(mock_collection, mock_config)

            # Verify the error was logged
            assert mock_stderr.write.called
            assert "not found in registry" in "".join(
                [c[0][0] for c in mock_stderr.write.call_args_list]
            )

            # Verify fallback to NaiveReranker happened
            assert mock_naive.called

            # Check the result contains the expected files
            assert result == ["file1.py", "file2.py"]


@pytest.mark.asyncio
async def test_get_query_result_files_general_exception(mock_collection, mock_config):
    """Test handling of unexpected exceptions during reranker loading."""
    # Set a custom reranker to trigger the import path
    mock_config.reranker = "buggy-reranker"

    # Create a patching context that raises an unexpected exception
    with patch("vectorcode.rerankers", new=MagicMock()) as mock_rerankers:
        # Configure the mock to raise RuntimeError when create_reranker is called
        mock_rerankers.create_reranker.side_effect = RuntimeError("Unexpected error")

        # Mock stderr to capture error messages
        with patch("sys.stderr") as mock_stderr:
            # Mock the NaiveReranker for fallback
            with patch(
                "vectorcode.subcommands.query.reranker.NaiveReranker"
            ) as mock_naive:
                mock_reranker_instance = MagicMock()
                mock_reranker_instance.rerank.return_value = ["file1.py", "file2.py"]
                mock_naive.return_value = mock_reranker_instance

                # This should catch the exception and fall back to NaiveReranker
                result = await get_query_result_files(mock_collection, mock_config)

                # Verify the error was logged
                assert mock_stderr.write.called

                # Verify fallback to NaiveReranker happened
                assert mock_naive.called

                # Check the result contains the expected files
                assert result == ["file1.py", "file2.py"]


@pytest.mark.asyncio
async def test_get_query_result_files_cross_encoder_error(mock_collection, mock_config):
    """Test the CrossEncoder special case with error handling."""
    # Set a cross encoder model to trigger that code path
    mock_config.reranker = "cross-encoder/model-name"

    # Mock the CrossEncoderReranker to raise an exception
    with patch(
        "vectorcode.subcommands.query.reranker.CrossEncoderReranker"
    ) as mock_cross_encoder:
        mock_cross_encoder.side_effect = ValueError("Model not found")

        # Mock stderr to capture error messages
        with patch("sys.stderr") as mock_stderr:
            # Mock the NaiveReranker for fallback
            with patch(
                "vectorcode.subcommands.query.reranker.NaiveReranker"
            ) as mock_naive:
                mock_reranker_instance = MagicMock()
                mock_reranker_instance.rerank.return_value = ["file1.py", "file2.py"]
                mock_naive.return_value = mock_reranker_instance

                # This should catch the exception and fall back
                result = await get_query_result_files(mock_collection, mock_config)

                # Verify the error was logged
                assert mock_stderr.write.called

                # Verify fallback to NaiveReranker happened
                assert mock_naive.called

                # Check the result contains the expected files
                assert result == ["file1.py", "file2.py"]
