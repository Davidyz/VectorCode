import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from chromadb import EmbeddingFunction
from numpy.typing import NDArray

from vectorcode.chunking import TreeSitterChunker
from vectorcode.cli_utils import Config
from vectorcode.database.types import (
    CollectionContent,
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
        self, collection_path: str, what: ResultType = ResultType.chunk
    ) -> int:
        """Returns the chunk count or file count of the given collection, depending on the value passed for `what`."""
        collection_content = await self.list(collection_path, what)
        match what:
            case ResultType.chunk:
                return len(collection_content.chunks)
            case ResultType.document:
                return len(collection_content.files)

    @abstractmethod
    async def query(
        self,
        collection_path: str,
        keywords_embeddings: list[NDArray],
        opts: QueryOpts,
    ) -> Sequence[QueryResult]:
        pass

    @abstractmethod
    async def vectorise(
        self,
        collection_path: str,
        file_path: str,
        chunker: TreeSitterChunker | None = None,
        embedding_function: EmbeddingFunction | None = None,
    ) -> VectoriseStats:
        """
        Vectorise the given file and add it to the database.
        The duplicate checking (using file hash) should be done outside of this function.
        """
        pass

    @abstractmethod
    async def list_collections(self) -> Sequence[CollectionInfo]:
        pass

    @abstractmethod
    async def list(
        self, collection_path: str, what: Optional[ResultType] = None
    ) -> CollectionContent:
        """
        When `what` is None, this method should populate both `CollectionContent.files` and `CollectionContent.chunks`.
        Otherwise, this method may populate only one of them to save waiting time.
        """
        pass

    @abstractmethod
    async def delete(self, collection_path: str, file_path: str | Sequence[str]):
        pass

    @abstractmethod
    async def drop(self, collection_path: str):
        """
        Delete a collection from the database.
        """
        pass
