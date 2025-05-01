import asyncio
import logging
from typing import Any

from vectorcode.cli_utils import Config

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
        from sentence_transformers import CrossEncoder

        if configs.reranker_params.get("model_name_or_path") is None:
            logger.warning(
                "'model_name_or_path' is not set. Fallback to 'cross-encoder/ms-marco-MiniLM-L-6-v2'"
            )
            configs.reranker_params["model_name_or_path"] = (
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        model_name = configs.reranker_params.pop("model_name_or_path")
        self.model = CrossEncoder(model_name, **configs.reranker_params)

    async def compute_similarity(self, results: list[str], query_message: str):
        scores = await asyncio.to_thread(
            self.model.predict, [(chunk, query_message) for chunk in results]
        )
        return list(float(i) for i in scores)
