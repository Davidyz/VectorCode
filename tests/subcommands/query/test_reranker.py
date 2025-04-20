from unittest.mock import MagicMock, patch

import pytest

from vectorcode.cli_utils import Config, QueryInclude
from vectorcode.subcommands.query.reranker import (
    CrossEncoderReranker,
    NaiveReranker,
    RerankerBase,
    __supported_rerankers,
    add_reranker,
    get_available_rerankers,
    get_reranker,
)


@pytest.fixture(scope="function")
def config():
    return Config(
        n_result=3,
        reranker_params={
            "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "device": "cpu",
        },
        reranker="CrossEncoderReranker",
        query=["query chunk 1", "query chunk 2"],
    )


@pytest.fixture(scope="function")
def naive_reranker_conf():
    return Config(
        n_result=3, reranker="NaiveReranker", query=["query chunk 1", "query chunk 2"]
    )


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def query_chunks():
    return ["query chunk 1", "query chunk 2"]


def test_reranker_base_method_is_abstract(config):
    with pytest.raises((NotImplementedError, TypeError)):
        RerankerBase(config)


def test_naive_reranker_initialization(naive_reranker_conf):
    """Test initialization of NaiveReranker"""
    reranker = NaiveReranker(naive_reranker_conf)
    assert reranker.n_result == 3


def test_reranker_create(naive_reranker_conf):
    reranker = NaiveReranker.create(naive_reranker_conf)
    assert isinstance(reranker, NaiveReranker)


def test_reranker_create_fail():
    class TestReranker(RerankerBase):
        def __init__(self, configs, **kwargs):
            raise Exception

    with pytest.raises(Exception):
        TestReranker.create(Config())


def test_naive_reranker_rerank(naive_reranker_conf, query_result):
    """Test basic reranking functionality of NaiveReranker"""
    reranker = NaiveReranker(naive_reranker_conf)
    result = reranker.rerank(query_result, ["foo", "bar"])

    # Check the result is a list of paths with correct length
    assert isinstance(result, list)
    assert len(result) <= naive_reranker_conf.n_result

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
    result = reranker.rerank(query_result_with_none, ["foo", "bar"])

    # Check the None path was handled without errors
    assert isinstance(result, list)
    # None should be filtered out
    assert None not in result


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_initialization(mock_cross_encoder: MagicMock, config):
    reranker = CrossEncoderReranker(config)

    # Verify constructor was called with correct parameters
    mock_cross_encoder.assert_called_once_with(**config.reranker_params)
    assert reranker.n_result == config.n_result


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_rerank(
    mock_cross_encoder, config, query_result, query_chunks
):
    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model

    # Configure mock rank method to return predetermined ranks
    mock_model.rank.return_value = [
        {"corpus_id": 0, "score": 0.9},
        {"corpus_id": 1, "score": 0.7},
        {"corpus_id": 2, "score": 0.8},
    ]

    reranker = CrossEncoderReranker(config)

    result = reranker.rerank(query_result, query_chunks)

    # Verify the model was called with correct parameters
    mock_model.rank.assert_called()

    # Check result
    assert isinstance(result, list)
    assert len(result) <= config.n_result

    # Check all returned items are strings (paths)
    for path in result:
        assert isinstance(path, str)


def test_naive_reranker_document_selection_logic(naive_reranker_conf):
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

    reranker = NaiveReranker(naive_reranker_conf)
    result = reranker.rerank(query_result, naive_reranker_conf.query)

    # Check that files are included (exact order depends on implementation details)
    assert len(result) > 0
    # Common files should be present
    assert "file2.py" in result or "file3.py" in result


def test_naive_reranker_with_chunk_ids(naive_reranker_conf):
    """Test NaiveReranker returns chunk IDs when QueryInclude.chunk is set"""
    naive_reranker_conf.include.append(
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
    reranker = NaiveReranker(naive_reranker_conf)
    result = reranker.rerank(query_result, naive_reranker_conf.query)

    assert isinstance(result, list)
    assert len(result) <= naive_reranker_conf.n_result
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
        config,
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
        },
        config.query,
    )

    assert isinstance(result, list)
    assert all(isinstance(id, str) for id in result)
    assert all(id in ["id1", "id2", "id3", "id4"] for id in result)


def test_get_reranker():
    config = Config(reranker="NaiveReranker")
    assert get_reranker(config).configs.reranker == "NaiveReranker"

    config = Config(reranker="CrossEncoderReranker", reranker_params={"device": "cpu"})
    reranker = get_reranker(config)
    assert reranker.configs.reranker == "CrossEncoderReranker"

    config = Config(
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_params={"device": "cpu"},
    )
    reranker = get_reranker(config)
    assert reranker.configs.reranker == "CrossEncoderReranker", (
        "configs.reranker should fallback to 'CrossEncoderReranker'"
    )
    assert (
        reranker.configs.reranker_params.get("model_name_or_path")
        == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ), "configs.reranker_params should fallback to default params."


def test_supported_rerankers_initialization(config, naive_reranker_conf):
    """Test that __supported_rerankers contains the expected default rerankers"""

    assert isinstance(get_reranker(config), CrossEncoderReranker)
    assert isinstance(get_reranker(naive_reranker_conf), NaiveReranker)
    assert len(get_available_rerankers()) == 2


def test_add_reranker_success():
    """Test successful registration of a new reranker"""

    original_count = len(get_available_rerankers())

    @add_reranker
    class TestReranker(RerankerBase):
        def rerank(self, results, query_chunks):
            return []

    assert len(get_available_rerankers()) == original_count + 1
    assert "TestReranker" in __supported_rerankers
    assert isinstance(get_reranker(Config(reranker="TestReranker")), TestReranker)
    __supported_rerankers.pop("TestReranker")


def test_add_reranker_duplicate():
    """Test duplicate reranker registration raises error"""

    # First registration should succeed
    @add_reranker
    class TestReranker(RerankerBase):
        def rerank(self, results, query_chunks):
            return []

    # Second registration should fail
    with pytest.raises(AttributeError):
        add_reranker(TestReranker)
    __supported_rerankers.pop("TestReranker")


def test_add_reranker_invalid_baseclass():
    """Test that non-RerankerBase classes can't be registered"""

    with pytest.raises(TypeError):

        @add_reranker
        class InvalidReranker:
            pass
