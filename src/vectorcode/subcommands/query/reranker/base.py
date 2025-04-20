import logging
from abc import ABC, abstractmethod
from typing import Any

from chromadb.api.types import QueryResult

from vectorcode.cli_utils import Config

logger = logging.getLogger(name=__name__)


class RerankerBase(ABC):
    """This is the base class for the rerankers.
    You should use the configs.reranker_params field to store and pass the parameters used for your reranker.
    You should implement the `rerank` method, which returns a list of chunk IDs if QueryInclude.chunk is in configs.include, or a list of paths otherwise.
    The items in the returned list should be sorted such that the relevance decreases along the list.

    The class doc string will be added to the error message if your reranker fails to initialise.
    Thus, this is a good place to put the instructions to configuring your reranker.
    """

    def __init__(self, configs: Config, **kwargs: Any):
        self.configs = configs
        self.n_result = configs.n_result

    @classmethod
    def create(cls, configs: Config, **kwargs: Any):
        try:
            return cls(configs, **kwargs)
        except Exception as e:
            e.add_note(
                "\n"
                + (
                    cls.__doc__
                    or f"There was an issue initialising {cls}. Please doublecheck your configuration."
                )
            )
            raise

    @abstractmethod
    def rerank(self, results: QueryResult, query_chunks: list[str]) -> list[str]:
        raise NotImplementedError
