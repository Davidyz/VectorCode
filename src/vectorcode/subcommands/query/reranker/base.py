import logging
from abc import abstractmethod
from typing import Any

from chromadb.api.types import QueryResult

from vectorcode.cli_utils import Config

logger = logging.getLogger(name=__name__)


class RerankerBase:
    def __init__(self, configs: Config, **kwargs: Any):
        self.configs = configs
        self.n_result = configs.n_result

    @abstractmethod
    def rerank(self, results: QueryResult, query_chunks: list[str]) -> list[str]:
        raise NotImplementedError
