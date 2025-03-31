import heapq
from abc import abstractmethod
from collections import defaultdict
from typing import Any, DefaultDict

import numpy
from chromadb.api.types import QueryResult

from vectorcode.cli_utils import Config, QueryInclude


class RerankerBase:
    def __init__(self, configs: Config, **kwargs: Any):
        self.configs = configs
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
            chunk_ids = results["ids"][query_chunk_idx]
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_distances = results["distances"][query_chunk_idx]
            # NOTE: distances, smaller is better.
            paths = [str(meta["path"]) for meta in chunk_metas]
            assert len(paths) == len(chunk_distances)
            for distance, identifier in zip(
                chunk_distances,
                chunk_ids if QueryInclude.chunk in self.configs.include else paths,
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


class CrossEncoderReranker(RerankerBase):
    def __init__(
        self, configs: Config, query_chunks: list[str], model_name: str, **kwargs: Any
    ):
        super().__init__(configs)
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, **kwargs)
        self.query_chunks = query_chunks

    def rerank(self, results: QueryResult) -> list[str]:
        assert results["metadatas"] is not None
        assert results["documents"] is not None
        documents: DefaultDict[str, list[float]] = defaultdict(list)
        for query_chunk_idx in range(len(self.query_chunks)):
            chunk_ids = results["ids"][query_chunk_idx]
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_docs = results["documents"][query_chunk_idx]
            ranks = self.model.rank(
                self.query_chunks[query_chunk_idx], chunk_docs, apply_softmax=True
            )
            for rank in ranks:
                if QueryInclude.chunk in self.configs.include:
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
