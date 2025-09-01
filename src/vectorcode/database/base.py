import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from vectorcode.chunking import Chunk
from vectorcode.cli_utils import Config
from vectorcode.database.types import (
    CollectionContent,
    CollectionID,
    CollectionInfo,
    QueryOpts,
    ResultType,
    VectoriseStats,
)
from vectorcode.subcommands.query.types import QueryResult

logger = logging.getLogger(name=__name__)


class DatabaseConnectorBase(ABC):  # pragma: nocover
    @classmethod
    def create(cls, configs: Config):
        try:
            return cls(configs)
        except Exception as e:  # pragma: nocover
            doc_string = cls.__doc__
            if doc_string:
                logger.warning(doc_string)
            raise e

    def __init__(self, configs: Config):
        """
        Use the `create` classmethod so that you get docs
        when something's wrong during the database initialisation.
        """
        self._configs = configs

    async def count(
        self, collection_id: CollectionID, what: ResultType = ResultType.chunk
    ) -> int:
        """Returns the chunk count or"""
        collection_content = await self.list(collection_id, what)
        match what:
            case ResultType.chunk:
                return len(collection_content.chunks)
            case ResultType.document:
                return len(collection_content.files)

    @abstractmethod
    async def query(
        self, collection_id: CollectionID, keywords: list[str], opts: QueryOpts
    ) -> Sequence[QueryResult]:
        pass

    @abstractmethod
    async def vectorise(
        self, collection_id: CollectionID, chunks: Sequence[Chunk]
    ) -> VectoriseStats:
        pass

    @abstractmethod
    async def list_collections(self) -> Sequence[CollectionInfo]:
        pass

    @abstractmethod
    async def list(
        self, collection_id: CollectionID, what: Optional[ResultType] = None
    ) -> CollectionContent:
        """
        When `what` is None, this method should populate both `CollectionContent.files` and `CollectionContent.chunks`.
        Otherwise, this method may populate only one of them to save waiting time.
        """
        pass
