import heapq
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy

from vectorcode.chunking import Chunk
from vectorcode.cli_utils import Config, QueryInclude
from vectorcode.subcommands.query.types import QueryResult

logger = logging.getLogger(name=__name__)


class RerankerBase(ABC):
    """This is the base class for the rerankers.
    You should use the configs.reranker_params field to store and pass the parameters used for your reranker.
    You should implement the `compute_similarity` method, which will be called by `rerank` to compute
    similarity scores between search query and results.
    The items in the returned list should be sorted such that the relevance decreases along the list.

    The class doc string will be added to the error message if your reranker fails to initialise.
    Thus, this is a good place to put the instructions to configuring your reranker.
    """

    def __init__(self, configs: Config, **kwargs: Any):
        self.configs = configs
        assert self.configs.query is not None, (
            "'configs' should contain the query messages."
        )
        self.n_result = configs.n_result
        self._raw_results: list[QueryResult] = []

    @classmethod
    def create(cls, configs: Config, **kwargs: Any):
        try:
            return cls(configs, **kwargs)
        except Exception as e:
            e.add_note(
                "\n"
                + (
                    cls.__doc__
                    or f"There was an issue initialising {cls}. Please doublecheck your configuration."
                )
            )
            raise

    @abstractmethod
    async def compute_similarity(
        self, results: list[QueryResult]
    ) -> None:  # pragma: nocover
        """
        Modify the `QueryResult.scores` field **IN-PLACE** so that they contain the correct scores.
        """
        raise NotImplementedError

    async def rerank(self, results: list[QueryResult]) -> list[str | Chunk]:
        if len(results) == 0:
            return []

        # compute the similarity scores
        await self.compute_similarity(results)

        # group the results by the query type: file (path) or chunk
        # and only keep the `top_k` results for each group
        group_by = "path"
        if QueryInclude.chunk in self.configs.include:
            group_by = "chunk"
        grouped_results = QueryResult.group(*results, by=group_by, top_k="auto")

        # compute the mean scores for each of the groups
        scores: dict[Chunk | str, float] = {}
        for key in grouped_results.keys():
            scores[key] = float(
                numpy.mean(tuple(i.mean_score() for i in grouped_results[key]))
            )

        return list(
            i
            for i in heapq.nlargest(
                self.configs.n_result, grouped_results.keys(), key=lambda x: scores[x]
            )
        )
