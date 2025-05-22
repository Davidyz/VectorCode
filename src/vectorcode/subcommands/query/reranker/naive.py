import logging
from typing import Any, Sequence

from vectorcode.cli_utils import Config

from .base import RerankerBase

logger = logging.getLogger(name=__name__)


class NaiveReranker(RerankerBase):
    """This reranker uses the distances between the embedding vectors in the database for the queries and the chunks as the measure of relevance.
    No special configs required.
    configs.reranker_params will be ignored.
    """

    def __init__(self, configs: Config, **kwargs: Any):
        super().__init__(configs)

    async def compute_similarity(
        self, results: list[str], query_message: str
    ) -> Sequence[float]:
        assert self._raw_results is not None, "Expecting raw results from the database."
        assert self._raw_results.get("distances") is not None
        assert self.configs.query, "Expecting query messages in self.configs"
        idx = self.configs.query.index(query_message)
        dist = self._raw_results.get("distances")
        if dist is None:  # pragma: nocover
            raise ValueError("QueryResult should contain distances!")
        else:
            return list(-i for i in dist[idx])
