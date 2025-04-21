import heapq
import logging
from collections import defaultdict
from typing import Any, DefaultDict

import numpy

from vectorcode.cli_utils import Config, QueryInclude

from .base import RerankerBase

logger = logging.getLogger(name=__name__)


class CrossEncoderReranker(RerankerBase):
    """This reranker uses [`CrossEncoder` from the sentence_transformers library](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html) for reranking.
    Parameters in configs.params will be passed to the `CrossEncoder` class in the `sentence_transformers` library.
    The default model is 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
    Consult sentence_transformers documentation for details on the available parameters.
    """

    def __init__(
        self,
        configs: Config,
        **kwargs: Any,
    ):
        super().__init__(configs)
        assert self.configs.query is not None, (
            "'configs' should contain the query messages."
        )
        from sentence_transformers import CrossEncoder

        if configs.reranker_params.get("model_name_or_path") is None:
            logger.warning(
                "'model_name_or_path' is not set. Fallback to 'cross-encoder/ms-marco-MiniLM-L-6-v2'"
            )
            configs.reranker_params["model_name_or_path"] = (
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        self.model = CrossEncoder(**configs.reranker_params)

    def rerank(self, results) -> list[str]:
        assert self.configs.query
        query_chunks = self.configs.query
        assert results["metadatas"] is not None
        assert results["documents"] is not None
        documents: DefaultDict[str, list[float]] = defaultdict(list)
        for query_chunk_idx in range(len(query_chunks)):
            chunk_ids = results["ids"][query_chunk_idx]
            chunk_metas = results["metadatas"][query_chunk_idx]
            chunk_docs = results["documents"][query_chunk_idx]
            ranks = self.model.rank(
                query_chunks[query_chunk_idx], chunk_docs, apply_softmax=True
            )
            for rank in ranks:
                if QueryInclude.chunk in self.configs.include:
                    documents[chunk_ids[rank["corpus_id"]]].append(float(rank["score"]))
                else:
                    documents[chunk_metas[rank["corpus_id"]]["path"]].append(
                        float(rank["score"])
                    )
        logger.debug("Document scores: %s", documents)
        top_k = int(numpy.mean(tuple(len(i) for i in documents.values())))
        for key in documents.keys():
            documents[key] = heapq.nlargest(top_k, documents[key])

        return heapq.nlargest(
            self.n_result,
            documents.keys(),
            key=lambda x: float(numpy.mean(documents[x])),
        )
