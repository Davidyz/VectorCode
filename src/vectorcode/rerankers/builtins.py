import heapq
from collections import defaultdict
from typing import DefaultDict, List

import numpy
from chromadb.api.types import QueryResult

from vectorcode.cli_utils import Config, QueryInclude

from .base import RerankerBase, register_reranker


@register_reranker("naive")
class NaiveReranker(RerankerBase):
    """A simple reranker that ranks documents by their mean distance."""

    def __init__(self, configs: Config = None, **kwargs):
        super().__init__(**kwargs)
        self.configs = configs
        self.n_result = configs.n_result if configs else kwargs.get("n_result", 10)

    def rerank(self, results: QueryResult) -> List[str]:
        """Rerank the query results by mean distance.

        Args:
            results: The query results from ChromaDB.

        Returns:
            A list of document IDs sorted by mean distance.
        """
        assert results["metadatas"] is not None
        assert results["distances"] is not None
        documents: DefaultDict[str, list[float]] = defaultdict(list)

        include = getattr(self.configs, "include", None) if self.configs else None

        for query_chunk_idx in range(len(results["ids"])):
            chunk_ids = results["ids"][query_chunk_idx]
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_distances = results["distances"][query_chunk_idx]
            # NOTE: distances, smaller is better.
            paths = [str(meta["path"]) for meta in chunk_metas]
            assert len(paths) == len(chunk_distances)
            for distance, identifier in zip(
                chunk_distances,
                chunk_ids if include and QueryInclude.chunk in include else paths,
            ):
                if identifier is None:
                    # so that vectorcode doesn't break on old collections.
                    continue
                documents[identifier].append(distance)

        top_k = int(numpy.mean(tuple(len(i) for i in documents.values())))
        for key in documents.keys():
            documents[key] = heapq.nsmallest(top_k, documents[key])

        return heapq.nsmallest(
            self.n_result, documents.keys(), lambda x: float(numpy.mean(documents[x]))
        )


@register_reranker("crossencoder")
class CrossEncoderReranker(RerankerBase):
    """A reranker that uses a cross-encoder model for reranking."""

    def __init__(
        self,
        configs: Config = None,
        query_chunks: List[str] = None,
        model_name: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.configs = configs
        self.n_result = configs.n_result if configs else kwargs.get("n_result", 10)

        # Handle model_name correctly
        self.model_name = model_name or kwargs.get("model_name")
        if not self.model_name:
            raise ValueError("model_name must be provided")

        self.query_chunks = query_chunks or kwargs.get("query_chunks", [])
        if not self.query_chunks:
            raise ValueError("query_chunks must be provided")

        # Import here to avoid requiring sentence-transformers for all rerankers
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(self.model_name, **kwargs)

    def rerank(self, results: QueryResult) -> List[str]:
        """Rerank the query results using a cross-encoder model.

        Args:
            results: The query results from ChromaDB.

        Returns:
            A list of document IDs sorted by cross-encoder scores.
        """
        assert results["metadatas"] is not None
        assert results["documents"] is not None
        documents: DefaultDict[str, list[float]] = defaultdict(list)

        include = getattr(self.configs, "include", None) if self.configs else None

        for query_chunk_idx in range(len(self.query_chunks)):
            chunk_ids = results["ids"][query_chunk_idx]
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_docs = results["documents"][query_chunk_idx]
            ranks = self.model.rank(
                self.query_chunks[query_chunk_idx], chunk_docs, apply_softmax=True
            )
            for rank in ranks:
                if include and QueryInclude.chunk in include:
                    documents[chunk_ids[rank["corpus_id"]]].append(float(rank["score"]))
                else:
                    documents[chunk_metas[rank["corpus_id"]]["path"]].append(
                        float(rank["score"])
                    )

        top_k = int(numpy.mean(tuple(len(i) for i in documents.values())))
        for key in documents.keys():
            documents[key] = heapq.nlargest(top_k, documents[key])

        return heapq.nlargest(
            self.n_result,
            documents.keys(),
            key=lambda x: float(numpy.mean(documents[x])),
        )
