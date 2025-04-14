"""Tests specifically targeting coverage gaps in the rerankers modules."""

import pytest

from vectorcode.cli_utils import Config
from vectorcode.rerankers import (
    CrossEncoderReranker,
    LlamaCppReranker,
    NaiveReranker,
    create_reranker,
    list_available_rerankers,
)


class TestRerankersCoverage:
    """Tests for coverage gaps in reranker modules."""

    def test_naive_reranker_none_path(self):
        """Test NaiveReranker handling of None paths in metadata."""
        # Create a config
        config = Config(n_result=2)

        # Create a reranker
        reranker = NaiveReranker(config)

        # Create results with a None path in metadata
        results = {
            "ids": [["id1", "id2", "id3"]],
            "metadatas": [
                [
                    {"path": "file1.py"},
                    {"path": None},  # None path here
                    {"path": "file3.py"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
            "documents": [["doc1", "doc2", "doc3"]],
        }

        # This should not raise any exceptions
        ranked_results = reranker.rerank(results)

        # Verify we get valid results (excluding the None path)
        assert len(ranked_results) <= 2  # n_result=2
        assert None not in ranked_results

    def test_create_reranker_not_found(self):
        """Test error handling when a reranker can't be found."""
        # Try to create a reranker with a name that doesn't exist
        with pytest.raises(ValueError) as exc_info:
            create_reranker("nonexistent-reranker")

        # Verify the error message includes available rerankers
        assert "not found in registry" in str(exc_info.value)
        assert "Available rerankers" in str(exc_info.value)

        # Available rerankers list should be included
        for reranker_name in list_available_rerankers():
            assert reranker_name in str(exc_info.value)

    def test_llama_cpp_reranker_empty_results(self):
        """Test LlamaCppReranker with empty results."""
        # Create the reranker
        reranker = LlamaCppReranker(model_name="test-model")

        # Mock empty results
        results = {"ids": [], "documents": []}

        # This should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            reranker.rerank(results)

    def test_llama_cpp_reranker_missing_fields(self):
        """Test LlamaCppReranker with missing fields in results."""
        # Create the reranker
        reranker = LlamaCppReranker(model_name="test-model")

        # Missing 'documents' field
        results = {
            "ids": [["id1", "id2"]],
            # documents field is missing
        }

        # This should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            reranker.rerank(results)

    def test_crossencoder_validation_error(self):
        """Test CrossEncoderReranker validation errors."""
        # Try to create a reranker without required parameters
        with pytest.raises(ValueError):
            CrossEncoderReranker()

        # Try with model_name but no query_chunks
        with pytest.raises(ValueError) as exc_info:
            CrossEncoderReranker(model_name="cross-encoder/model")
        assert "query_chunks must be provided" in str(exc_info.value)

        # Try with query_chunks but no model_name
        with pytest.raises(ValueError) as exc_info:
            CrossEncoderReranker(query_chunks=["query"])
        assert "model_name must be provided" in str(exc_info.value)
