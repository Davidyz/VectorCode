import heapq
import logging
from collections import defaultdict
from typing import Any, DefaultDict

import numpy

from vectorcode.cli_utils import Config, QueryInclude

from .base import RerankerBase

logger = logging.getLogger(name=__name__)


class NaiveReranker(RerankerBase):
    """This reranker uses the distances between the embedding vectors in the database for the queries and the chunks as the measure of relevance.
    No special configs required.
    configs.reranker_params will be ignored.
    """

    def __init__(self, configs: Config, **kwargs: Any):
        super().__init__(configs)

    def rerank(self, results) -> list[str]:
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
                if identifier is None:  # pragma: nocover
                    # so that vectorcode doesn't break on old collections.
                    continue
                documents[identifier].append(distance)
        logger.debug("Document scores: %s", documents)
        top_k = int(numpy.mean(tuple(len(i) for i in documents.values())))
        for key in documents.keys():
            documents[key] = heapq.nsmallest(top_k, documents[key])

        return heapq.nsmallest(
            self.n_result, documents.keys(), lambda x: float(numpy.mean(documents[x]))
        )
