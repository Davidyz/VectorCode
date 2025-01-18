import heapq
from abc import abstractmethod
from collections import defaultdict
from typing import Any, DefaultDict

import numpy
from chromadb.api.types import QueryResult

from vectorcode.cli_utils import Config


class RerankerBase:
    def __init__(self, configs: Config, **kwargs: Any):
        self.n_result = configs.n_result

    @abstractmethod
    def rerank(self, results: QueryResult) -> list[str]:
        raise NotImplementedError


class NaiveReranker(RerankerBase):
    def __init__(self, configs: Config, **kwargs: Any):
        super().__init__(configs)

    def rerank(self, results: QueryResult) -> list[str]:
        assert results["metadatas"] is not None
        assert results["distances"] is not None
        documents: DefaultDict[str, list[float]] = defaultdict(list)
        for query_chunk_idx in range(len(results["ids"])):
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_distances = results["distances"][query_chunk_idx]
            # NOTE: distances, smaller is better.
            paths = [str(meta["path"]) for meta in chunk_metas]
            assert len(paths) == len(chunk_distances)
            for distance, path in zip(chunk_distances, paths):
                documents[path].append(distance)

        return heapq.nsmallest(
            self.n_result, documents.keys(), lambda x: float(numpy.mean(documents[x]))
        )


class FlagEmbeddingReranker(RerankerBase):
    def __init__(
        self, configs: Config, chunks: list[str], model_name_or_path: str, **kwargs: Any
    ):
        super().__init__(configs)
        from FlagEmbedding import FlagAutoReranker

        self.original_chunks = chunks
        self.model = FlagAutoReranker.from_finetuned(
            model_name_or_path=model_name_or_path, **kwargs
        )
        self.query_chunks = chunks

    def rerank(self, results: QueryResult) -> list[str]:
        assert results["metadatas"] is not None
        assert results["documents"] is not None
        documents: DefaultDict[str, list[float]] = defaultdict(list)
        for query_chunk_idx in range(len(self.query_chunks)):
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_docs = results["documents"][query_chunk_idx]
            similarities = self.model.compute_score(
                [(self.query_chunks[query_chunk_idx], doc) for doc in chunk_docs],
                normalize=True,
            )
            for i, meta in enumerate(chunk_metas):
                documents[meta["path"]].append(similarities[i])
        return heapq.nlargest(
            self.n_result,
            documents.keys(),
            key=lambda x: float(numpy.mean(documents[x])),
        )
