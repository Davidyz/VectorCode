import pytest
from tree_sitter import Point

from vectorcode.chunking import Chunk
from vectorcode.subcommands.query.types import QueryResult


def make_dummy_chunk():
    return QueryResult(
        path="dummy1.py",
        chunk=Chunk(
            text="hello", start=Point(row=1, column=0), end=Point(row=1, column=4)
        ),
        query=["hello"],
        scores=[0.9],
    )


def test_QueryResult_merge():
    res1, res2 = (make_dummy_chunk(), make_dummy_chunk())
    res2.query = ["bye"]
    res2.scores = [0.1]

    merged = QueryResult.merge(res1, res2)
    assert merged.path == res1.path
    assert merged.chunk == res1.chunk
    assert merged.mean_score() == 0.5
    assert merged.query == ("hello", "bye")


def test_QueryResult_merge_failed():
    res1, res2 = (make_dummy_chunk(), make_dummy_chunk())
    res2.path = "dummy2.py"
    with pytest.raises(ValueError):
        QueryResult.merge(res1, res2)


def test_QueryResult_group_by_path():
    res1, res2 = (make_dummy_chunk(), make_dummy_chunk())
    res2.chunk = Chunk(
        "hello", start=Point(row=2, column=0), end=Point(row=2, column=4)
    )
    res2.query = ["bye"]
    res2.scores = [0.1]

    grouped_dict = QueryResult.group(res1, res2)
    assert len(grouped_dict.keys()) == 1
    assert len(grouped_dict["dummy1.py"]) == 2


def test_QueryResult_group_by_chunk():
    res1, res2 = (make_dummy_chunk(), make_dummy_chunk())
    res2.query = ["bye"]
    res2.scores = [0.1]

    grouped_dict = QueryResult.group(res1, res2, by="chunk")
    assert len(grouped_dict.keys()) == 1
    assert len(grouped_dict[res1.chunk]) == 2


def test_QueryResult_group_top_k():
    res1, res2 = (make_dummy_chunk(), make_dummy_chunk())
    res2.chunk = Chunk(
        "hello", start=Point(row=2, column=0), end=Point(row=2, column=4)
    )
    res2.query = ["bye"]
    res2.scores = [0.1]

    grouped_dict = QueryResult.group(res1, res2, top_k=1)
    assert len(grouped_dict.keys()) == 1
    assert len(grouped_dict["dummy1.py"]) == 1
    assert grouped_dict["dummy1.py"][0].query[0] == "hello"


def test_QueryResult_lt():
    res1, res2 = (make_dummy_chunk(), make_dummy_chunk())
    res2.chunk = Chunk(
        "hello", start=Point(row=2, column=0), end=Point(row=2, column=4)
    )
    res2.query = ["bye"]
    res2.scores = [0.1]
    assert res2 < res1
