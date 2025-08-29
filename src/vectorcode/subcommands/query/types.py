import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Union

import numpy

from vectorcode.chunking import Chunk


@dataclass
class QueryResult:
    """
    The container for one single query result.

    args:
    - path: path to the file
    - content: `vectorcode.chunking.Chunk` object that stores the chunk
    - query: query messages used for the search
    - scores: similarity scores for the corresponding query.
    """

    path: str
    chunk: Chunk
    query: tuple[str, ...]
    scores: tuple[float, ...]

    @classmethod
    def merge(cls, *results: "QueryResult") -> "QueryResult":
        """
        Given the results of a single chunk/document from different queries, merge them into a single `QueryResult` object.
        """
        for i in range(len(results) - 1):
            if (i < len(results) - 1) and not results[i].is_same_doc(results[i + 1]):
                raise ValueError(
                    f"The inputs are not the same chunk: {results[i]}, {results[i + 1]}"
                )

        return QueryResult(
            path=results[0].path,
            chunk=results[0].chunk,
            query=sum((tuple(i.query) for i in results), start=tuple()),
            scores=sum((tuple(i.scores) for i in results), start=tuple()),
        )

    @staticmethod
    def group(
        *results: "QueryResult",
        by: Union[Literal["path"], Literal["chunk"]] = "path",
        top_k: int | Literal["auto"] | None = None,
    ) -> dict[Chunk | str, list["QueryResult"]]:
        """
        Group the query results based on `key`.

        args:
        - `by`: either "path" or "chunk"
        - `top_k`: if set, only return the top k results for each group based on mean scores. If "auto", top k is decided by the mean number of results per group.

        returns:
        - a dictionary that maps either path or chunk to a list of `QueryResult` object.

        """
        assert by in {"path", "chunk"}
        grouped_result: dict[Chunk | str, list["QueryResult"]] = defaultdict(list)

        for res in results:
            grouped_result[getattr(res, by)].append(res)

        if top_k == "auto":
            top_k = int(numpy.mean(tuple(len(i) for i in grouped_result.values())))

        if top_k and top_k > 0:
            for group in grouped_result.keys():
                grouped_result[group] = heapq.nlargest(top_k, grouped_result[group])
        return grouped_result

    def mean_score(self):
        return float(numpy.mean(self.scores))

    def __lt__(self, other: "QueryResult"):
        assert isinstance(other, QueryResult)
        return self.mean_score() < other.mean_score()

    def __gt__(self, other: "QueryResult"):
        assert isinstance(other, QueryResult)
        return self.mean_score() > other.mean_score()

    def __eq__(self, other: object, /) -> bool:
        return (
            isinstance(other, QueryResult) and self.mean_score() == other.mean_score()
        )

    def is_same_doc(self, other: "QueryResult") -> bool:
        return self.path == other.path and self.chunk == other.chunk
