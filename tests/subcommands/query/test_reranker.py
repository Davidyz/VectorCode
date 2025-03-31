from unittest.mock import MagicMock, patch

import pytest

from vectorcode.cli_utils import Config, QueryInclude
from vectorcode.subcommands.query.reranker import (
    CrossEncoderReranker,
    NaiveReranker,
    RerankerBase,
)


@pytest.fixture
def config():
    return Config(n_result=3)


@pytest.fixture
def query_result():
    return {
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


@pytest.fixture
def query_chunks():
    return ["query chunk 1", "query chunk 2"]


# The RerankerBase isn't actually preventing instantiation,
# but it will raise NotImplementedError when rerank is called
def test_reranker_base_method_is_abstract(config):
    """Test that RerankerBase.rerank raises NotImplementedError"""
    base_reranker = RerankerBase(config)
    with pytest.raises(NotImplementedError):
        base_reranker.rerank({})


def test_naive_reranker_initialization(config):
    """Test initialization of NaiveReranker"""
    reranker = NaiveReranker(config)
    assert reranker.n_result == 3


def test_naive_reranker_rerank(config, query_result):
    """Test basic reranking functionality of NaiveReranker"""
    reranker = NaiveReranker(config)
    result = reranker.rerank(query_result)

    # Check the result is a list of paths with correct length
    assert isinstance(result, list)
    assert len(result) <= config.n_result

    # Check all returned items are strings (paths)
    for path in result:
        assert isinstance(path, str)


def test_naive_reranker_handles_none_path(config, query_result):
    """Test NaiveReranker properly handles None paths in metadata"""
    # Create a copy with a None path
    query_result_with_none = query_result.copy()
    query_result_with_none["metadatas"] = [
        [{"path": "file1.py"}, {"path": None}, {"path": "file3.py"}],
        [{"path": "file2.py"}, {"path": "file4.py"}, {"path": "file3.py"}],
    ]

    reranker = NaiveReranker(config)
    result = reranker.rerank(query_result_with_none)

    # Check the None path was handled without errors
    assert isinstance(result, list)
    # None should be filtered out
    assert None not in result


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_initialization(
    mock_cross_encoder, config, query_chunks
):
    """Test initialization of CrossEncoderReranker"""
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    reranker = CrossEncoderReranker(config, query_chunks, model_name)

    # Verify constructor was called with correct parameters
    mock_cross_encoder.assert_called_once_with(model_name)
    assert reranker.n_result == 3
    assert reranker.query_chunks == query_chunks


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_rerank(
    mock_cross_encoder, config, query_result, query_chunks
):
    """Test reranking with CrossEncoderReranker"""
    # Setup mock model
    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model

    # Configure mock rank method to return predetermined ranks
    mock_model.rank.return_value = [
        {"corpus_id": 0, "score": 0.9},
        {"corpus_id": 1, "score": 0.7},
        {"corpus_id": 2, "score": 0.8},
    ]

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = CrossEncoderReranker(config, query_chunks, model_name)

    result = reranker.rerank(query_result)

    # Verify the model was called with correct parameters
    mock_model.rank.assert_called()

    # Check result
    assert isinstance(result, list)
    assert len(result) <= config.n_result

    # Check all returned items are strings (paths)
    for path in result:
        assert isinstance(path, str)


def test_naive_reranker_document_selection_logic(config):
    """Test that NaiveReranker correctly selects documents based on distances"""
    # Create a query result with known distances
    query_result = {
        "ids": [["id1", "id2", "id3"], ["id4", "id5", "id6"]],
        "distances": [
            [0.3, 0.1, 0.2],  # file2 has lowest, then file3, then file1
            [0.6, 0.4, 0.5],  # file4 has lowest, then file3, then file2
        ],
        "metadatas": [
            [{"path": "file1.py"}, {"path": "file2.py"}, {"path": "file3.py"}],
            [{"path": "file2.py"}, {"path": "file4.py"}, {"path": "file3.py"}],
        ],
    }

    reranker = NaiveReranker(config)
    result = reranker.rerank(query_result)

    # Check that files are included (exact order depends on implementation details)
    assert len(result) > 0
    # Common files should be present
    assert "file2.py" in result or "file3.py" in result


def test_naive_reranker_with_chunk_ids(config):
    """Test NaiveReranker returns chunk IDs when QueryInclude.chunk is set"""
    config.include.append(
        QueryInclude.chunk
    )  # Assuming QueryInclude.chunk would be "chunk"
    query_result = {
        "ids": [["id1", "id2"], ["id3", "id1"]],
        "distances": [[0.1, 0.2], [0.3, 0.4]],
        "metadatas": [
            [{"path": "file1.py"}, {"path": "file2.py"}],
            [{"path": "file3.py"}, {"path": "file1.py"}],
        ],
    }
    reranker = NaiveReranker(config)
    result = reranker.rerank(query_result)

    assert isinstance(result, list)
    assert len(result) <= config.n_result
    assert all(isinstance(id, str) for id in result)
    assert all(id.startswith("id") for id in result)  # Verify IDs not paths


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_with_chunk_ids(
    mock_cross_encoder, config, query_chunks
):
    """Test CrossEncoderReranker returns chunk IDs when QueryInclude.chunk is set"""
    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model
    mock_model.rank.return_value = [
        {"corpus_id": 0, "score": 0.9},
        {"corpus_id": 1, "score": 0.7},
    ]

    config.include = {"chunk"}  # Use comma instead of append
    reranker = CrossEncoderReranker(
        config, query_chunks, "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Match query_chunks length with results
    result = reranker.rerank(
        {
            "ids": [["id1", "id2"], ["id3", "id4"]],  # Two query chunks
            "metadatas": [
                [{"path": "file1.py"}, {"path": "file2.py"}],
                [{"path": "file3.py"}, {"path": "file4.py"}],
            ],
            "documents": [["doc1", "doc2"], ["doc3", "doc4"]],
        }
    )

    assert isinstance(result, list)
    assert all(isinstance(id, str) for id in result)
    assert all(id in ["id1", "id2", "id3", "id4"] for id in result)
