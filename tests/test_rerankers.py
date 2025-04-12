import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest

from vectorcode.rerankers import (
    LlamaCppReranker,
    NaiveReranker,
    RerankerBase,
    create_reranker,
    get_reranker_class,
    list_available_rerankers,
    register_reranker,
)


class TestRerankers(unittest.TestCase):
    def setUp(self):
        # Create a simple query result for testing
        self.query_result = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [
                [
                    {"path": "path1"},
                    {"path": "path2"},
                    {"path": "path3"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        # Create a mock config
        self.mock_config = Mock()
        self.mock_config.n_result = 2
        self.mock_config.include = []

    def test_base_reranker(self):
        """Test that RerankerBase is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            # Should raise TypeError because rerank is abstract
            RerankerBase()

    def test_naive_reranker(self):
        """Test the NaiveReranker implementation."""
        reranker = NaiveReranker(configs=self.mock_config)
        results = reranker.rerank(self.query_result)

        # Should return a list of document IDs
        assert isinstance(results, list)
        # Should return n_result items
        assert len(results) == self.mock_config.n_result
        # Should contain paths from metadatas
        assert all(item in ["path1", "path2", "path3"] for item in results)

    def test_llama_cpp_reranker(self):
        """Test the LlamaCppReranker implementation."""
        # Test initialization with model_name as positional arg
        reranker1 = LlamaCppReranker("test_model")
        assert reranker1.api_url == "test_model"

        # Test initialization with model_name as keyword arg
        reranker2 = LlamaCppReranker(model_name="test_model2")
        assert reranker2.api_url == "test_model2"

        # Test initialization with no model_name (should use default)
        with patch.dict(os.environ, {"VECTORCODE_RERANKING_API_URL": "env_test_url"}):
            reranker3 = LlamaCppReranker()
            assert reranker3.api_url == "env_test_url"

        # Test rerank method raises NotImplementedError
        with pytest.raises(NotImplementedError):
            reranker1.rerank(self.query_result)

    def test_registry(self):
        """Test the reranker registry functionality."""
        # Test listing available rerankers
        rerankers = list_available_rerankers()
        assert "naive" in rerankers
        assert "crossencoder" in rerankers
        assert "llamacpp" in rerankers

        # Test getting a reranker class
        naive_class = get_reranker_class("naive")
        assert naive_class == NaiveReranker

        # Test registering a new reranker
        @register_reranker("test_reranker")
        class TestReranker(RerankerBase):
            def rerank(self, results):
                return ["test1", "test2"]

        # Check that it's properly registered
        assert "test_reranker" in list_available_rerankers()
        assert get_reranker_class("test_reranker") == TestReranker

    def test_create_reranker(self):
        """Test the create_reranker factory function."""
        # Test creating a NaiveReranker
        reranker1 = create_reranker("naive", configs=self.mock_config)
        assert isinstance(reranker1, NaiveReranker)

        # Test using legacy name
        reranker2 = create_reranker("NaiveReranker", configs=self.mock_config)
        assert isinstance(reranker2, NaiveReranker)

        # Test with invalid name
        with pytest.raises(ValueError):
            create_reranker("invalid_reranker", configs=self.mock_config)

    def test_dynamic_loading(self):
        """Test dynamic loading of custom rerankers."""
        # Create a temporary reranker module
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as f:
            f.write("""
from vectorcode.rerankers import RerankerBase, register_reranker

@register_reranker("custom_test")
class CustomTestReranker(RerankerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def rerank(self, results):
        return ["custom1", "custom2"]
""")
            f.flush()

            # Add its directory to Python path
            sys.path.append(os.path.dirname(f.name))

            try:
                # Import the module
                module_name = os.path.basename(f.name)[:-3]  # Remove .py
                __import__(module_name)

                # Now test creating the custom reranker
                reranker = create_reranker("custom_test")

                # Check that it works
                assert reranker.rerank({}) == ["custom1", "custom2"]
            finally:
                # Clean up
                sys.path.remove(os.path.dirname(f.name))
                if module_name in sys.modules:
                    del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main()
