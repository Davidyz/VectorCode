import logging
from typing import Any

from vectorcode.cli_utils import Config
from vectorcode.database.types import QueryResult

from .base import RerankerBase

logger = logging.getLogger(name=__name__)


class NaiveReranker(RerankerBase):
    """This reranker uses the distances between the embedding vectors in the database for the queries and the chunks as the measure of relevance.
    No special configs required.
    configs.reranker_params will be ignored.
    """

    def __init__(self, configs: Config, **kwargs: Any):
        super().__init__(configs)

    async def compute_similarity(self, results: list[QueryResult]):
        """
        Do nothing, because the QueryResult objects already contain distances.
        """
        pass
