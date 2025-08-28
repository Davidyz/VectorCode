from typing import cast
from unittest.mock import MagicMock, patch

import numpy
import pytest

from vectorcode.cli_utils import Config
from vectorcode.subcommands.query.reranker import (
    CrossEncoderReranker,
    NaiveReranker,
    RerankerBase,
    __supported_rerankers,
    add_reranker,
    get_available_rerankers,
    get_reranker,
)
from vectorcode.subcommands.query.types import QueryResult


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
def query_result() -> list[QueryResult]:
    return [
        QueryResult(
            path="file1.py",
            chunk=MagicMock(),
            query=("query chunk 1",),
            scores=(0.5,),
        ),
        QueryResult(
            path="file2.py",
            chunk=MagicMock(),
            query=("query chunk 1",),
            scores=(0.9,),
        ),
        QueryResult(
            path="file3.py",
            chunk=MagicMock(),
            query=("query chunk 1",),
            scores=(0.3,),
        ),
        QueryResult(
            path="file2.py",
            chunk=MagicMock(),
            query=("query chunk 2",),
            scores=(0.6,),
        ),
        QueryResult(
            path="file4.py",
            chunk=MagicMock(),
            query=("query chunk 2",),
            scores=(0.7,),
        ),
        QueryResult(
            path="file3.py",
            chunk=MagicMock(),
            query=("query chunk 2",),
            scores=(0.2,),
        ),
    ]


@pytest.fixture(scope="function")
def empty_query_result():
    return []


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


@pytest.mark.asyncio
async def test_naive_reranker_rerank(naive_reranker_conf, query_result):
    """Test basic reranking functionality of NaiveReranker"""
    reranker = NaiveReranker(naive_reranker_conf)
    result = await reranker.rerank(query_result)

    # Check the result is a list of paths with correct length
    assert isinstance(result, list)
    assert len(result) <= naive_reranker_conf.n_result

    # Check all returned items are strings (paths)
    for res in result:
        assert isinstance(res, str)


@pytest.mark.asyncio
async def test_naive_reranker_rerank_empty_result(
    naive_reranker_conf, empty_query_result
):
    reranker = NaiveReranker(naive_reranker_conf)
    result = await reranker.rerank(empty_query_result)
    assert len(result) == 0


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_initialization(mock_cross_encoder: MagicMock, config):
    model_name = config.reranker_params["model_name_or_path"]
    reranker = CrossEncoderReranker(config)
    # Verify constructor was called with correct parameters
    mock_cross_encoder.assert_called_once_with(model_name, **config.reranker_params)
    assert reranker.n_result == config.n_result


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_initialization_fallback_model_name(
    mock_cross_encoder: MagicMock, config
):
    config.reranker_params = {}
    reranker = CrossEncoderReranker(config)

    # Verify constructor was called with correct parameters
    mock_cross_encoder.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert reranker.n_result == config.n_result


@pytest.mark.asyncio
@patch("sentence_transformers.CrossEncoder")
async def test_cross_encoder_reranker_rerank(mock_cross_encoder, config, query_result):
    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model

    mock_model.predict = lambda x: numpy.random.random((len(x),))

    reranker = CrossEncoderReranker(config)
    result = await reranker.rerank(query_result)

    # Result assertions
    assert isinstance(result, list)
    assert all(isinstance(path, str) for path in result)
    assert len(result) <= config.n_result


@pytest.mark.asyncio
async def test_naive_reranker_document_selection_logic(
    naive_reranker_conf, query_result
):
    """Test that NaiveReranker correctly selects documents based on distances"""
    # Create a query result with known distances

    reranker = NaiveReranker(naive_reranker_conf)
    result = await reranker.rerank(query_result)

    # Check that files are included (exact order depends on implementation details)
    assert len(result) > 0
    # Common files should be present
    assert "file2.py" in result or "file3.py" in result


def test_get_reranker(config, naive_reranker_conf):
    assert get_reranker(naive_reranker_conf).configs.reranker == "NaiveReranker"

    reranker = get_reranker(config)
    assert reranker.configs.reranker == "CrossEncoderReranker"

    reranker = cast(CrossEncoderReranker, get_reranker(config))
    assert reranker.configs.reranker == "CrossEncoderReranker", (
        "configs.reranker should fallback to 'CrossEncoderReranker'"
    )


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
        async def compute_similarity(self, results, query_message):
            return []

    assert len(get_available_rerankers()) == original_count + 1
    assert "TestReranker" in __supported_rerankers
    assert isinstance(
        get_reranker(Config(reranker="TestReranker", query=["hello world"])),
        TestReranker,
    )
    __supported_rerankers.pop("TestReranker")


def test_add_reranker_duplicate():
    """Test duplicate reranker registration raises error"""

    # First registration should succeed
    @add_reranker
    class TestReranker(RerankerBase):
        async def compute_similarity(self, results, query_message):
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
