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


class ArithmeticMeanReranker(RerankerBase):
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

        docs = sorted(
            [key for key in documents.keys()],
            key=lambda x: float(numpy.mean(documents[x])),
        )
        return docs[: self.n_result]


class FlagEmbeddingReranker(RerankerBase):
    def __init__(
        self, configs: Config, chunks: list[str], model_name_or_path: str, **kwargs: Any
    ):
        super().__init__(configs)
        from FlagEmbedding import FlagAutoModel

        self.original_chunks = chunks
        self.model = FlagAutoModel.from_finetuned(
            model_name_or_path=model_name_or_path, **kwargs
        )
        self.query_embeddings = self.model.encode(self.original_chunks)

    def rerank(self, results: QueryResult) -> list[str]:
        assert results["metadatas"] is not None
        assert results["distances"] is not None
        paths = set()
        for metas in results["metadatas"]:
            for meta in metas:
                paths.add(meta.get("path"))
        for query_chunk_idx in range(len(results["ids"])):
            # TODO: implement a vectorised way to compute the similarities in batch.
            # ref: https://github.com/FlagOpen/FlagEmbedding?tab=readme-ov-file#quick-start
            pass
