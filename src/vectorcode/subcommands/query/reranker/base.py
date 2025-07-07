import heapq
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, DefaultDict, Optional, Sequence, cast

import numpy
from chromadb.api.types import QueryResult

from vectorcode.cli_utils import Config, QueryInclude

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
        self._raw_results: Optional[QueryResult] = None

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
        self, results: list[str], query_message: str
    ) -> Sequence[float]:  # pragma: nocover
        """Given a list of n results and 1 query message,
        return a list-like object of length n that contains the similarity scores between
        each item in `results` and the `query_message`.

        A high similarity score means the strings are semantically similar to each other.
        `query_message` will be loaded in the same order as they appear in `self.configs.query`.

        If you need the raw query results from chromadb,
        it'll be saved in `self._raw_results` before this method is called.
        """
        raise NotImplementedError

    async def rerank(self, results: QueryResult | dict) -> list[str]:
        if len(results["ids"]) == 0 or all(len(i) == 0 for i in results["ids"]):
            return []

        self._raw_results = cast(QueryResult, results)
        query_chunks = self.configs.query
        assert query_chunks
        assert results["metadatas"] is not None
        assert results["documents"] is not None
        documents: DefaultDict[str, list[float]] = defaultdict(list)
        for query_chunk_idx in range(len(query_chunks)):
            chunk_ids = results["ids"][query_chunk_idx]
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_docs = results["documents"][query_chunk_idx]
            scores = await self.compute_similarity(
                chunk_docs, query_chunks[query_chunk_idx]
            )
            for i, score in enumerate(scores):
                if QueryInclude.chunk in self.configs.include:
                    documents[chunk_ids[i]].append(float(score))
                else:
                    documents[str(chunk_metas[i]["path"])].append(float(score))

        logger.debug("Document scores: %s", documents)
        top_k = int(numpy.mean(tuple(len(i) for i in documents.values())))
        for key in documents.keys():
            documents[key] = heapq.nlargest(top_k, documents[key])

        self._raw_results = None

        return heapq.nlargest(
            self.n_result,
            documents.keys(),
            key=lambda x: float(numpy.mean(documents[x])),
        )
