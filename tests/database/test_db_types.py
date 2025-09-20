from vectorcode.chunking import Chunk
from vectorcode.database.types import QueryResult, VectoriseStats

"""
For boilerplate code that wasn't covered in other tests.
"""


def test_vectorstats_add():
    assert VectoriseStats(
        add=1, update=2, removed=3, skipped=4, failed=5
    ) + VectoriseStats(
        add=5, update=4, removed=3, skipped=2, failed=1
    ) == VectoriseStats(add=6, update=6, removed=6, skipped=6, failed=6)

    assert VectoriseStats(
        add=1, update=2, removed=3, skipped=4, failed=5
    ) + VectoriseStats(
        add=5, update=4, removed=3, skipped=2, failed=1
    ) != VectoriseStats(add=6, update=6, removed=6, skipped=6, failed=5)


def test_query_result_equal():
    assert QueryResult(
        path="some_path",
        chunk=Chunk(text="some_text"),
        query=("some_query",),
        scores=(1,),
    ) == QueryResult(
        path="other_path",
        chunk=Chunk(text="other_text"),
        query=("some_query",),
        scores=(1.0,),
    )

    assert QueryResult(
        path="some_path",
        chunk=Chunk(text="some_text"),
        query=("some_query",),
        scores=(1,),
    ) != QueryResult(
        path="other_path",
        chunk=Chunk(text="other_text"),
        query=("some_query",),
        scores=(2.0,),
    )
