import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Self, Sequence

from chromadb import EmbeddingFunction
from numpy.typing import NDArray

from vectorcode.chunking import TreeSitterChunker
from vectorcode.cli_utils import Config
from vectorcode.common import get_embedding_function
from vectorcode.database.types import (
    CollectionContent,
    CollectionInfo,
    ResultType,
    VectoriseStats,
)
from vectorcode.subcommands.query.types import QueryResult

logger = logging.getLogger(name=__name__)


"""
For developers:

To implement a custom database connector, you should inherit the following 
`DatabaseConnectorBase` class and implement ALL abstract methods. 

You should also try to wrap the exceptions with the ones in 
`src/vectorcode/database/errors.py` where appropriate, because this helps the 
CLI/LSP/MCP interfaces to handle some common edge cases. To do this, you should do the following in a try-except block:
```python
from vectorcode.database.errors import CollectionNotFoundError

try:
    some_action_here()
except SomeCustomException as e:
    raise CollectionNotFoundError("The collection was not found.") from e
```
This will preserve the correct call stack in the error message and makes debugging 
easier.
"""


class DatabaseConnectorBase(ABC):  # pragma: nocover
    @classmethod
    def create(cls, configs: Config):
        try:
            return cls(configs)
        except Exception as e:  # pragma: nocover
            doc_string = cls.__doc__
            if doc_string:
                e.add_note(doc_string)
            raise

    def __init__(self, configs: Config):
        """
        Use the `create` classmethod so that you get docs
        when something's wrong during the database initialisation.
        """
        self._configs = configs

    async def count(self, what: ResultType = ResultType.chunk) -> int:
        """
        Returns the chunk count or file count of the given collection, depending on the value passed for `what`.
        """
        collection_content = await self.list_collection_content(what)
        match what:
            case ResultType.chunk:
                return len(collection_content.chunks)
            case ResultType.document:
                return len(collection_content.files)

    @abstractmethod
    async def query(
        self,
    ) -> Sequence[QueryResult]:
        pass

    @abstractmethod
    async def vectorise(
        self,
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
        """
        List collections in the database.
        """
        pass

    @abstractmethod
    async def list_collection_content(
        self, what: Optional[ResultType] = None
    ) -> CollectionContent:
        """
        List the content of a collection (from `self._configs.project_root`).

        When `what` is None, this method should populate both `CollectionContent.files` and `CollectionContent.chunks`.
        Otherwise, this method may populate only one of them to save waiting time.
        """
        pass

    @abstractmethod
    async def delete(
        self,
    ) -> int:
        """
        Delete files from the database (doesn't remove files on disk).
        Returns the actual number of files deleted.
        """
        pass

    @abstractmethod
    async def drop(self):
        """
        Delete a collection (`self._configs.project_root`) from the database.
        """
        pass

    def _check_new_config(self, new_config: Config) -> bool:
        """
        Verify that the `new_config` is a valid one for updating.
        """
        assert isinstance(new_config, Config), "`new_config` is not a `Config` object."
        return (
            new_config.db_type == self._configs.db_type
            and new_config.db_params == self._configs.db_params
        )

    def update_config(self, new_config: Config) -> Self:
        assert self._check_new_config(new_config), (
            "The new config has different database configs."
        )

        # no need to make this one async
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._configs = loop.run_until_complete(self._configs.merge_from(new_config))
        return self

    def replace_config(self, new_config: Config) -> Self:
        assert self._check_new_config(new_config), (
            "The new config has different database configs."
        )
        self._configs = new_config
        return self

    async def check_orphanes(self) -> int:
        """
        Check for files that are in the database, but no longer on the disk, and remove them.
        """

        orphanes: list[str] = []
        database_files = (await self.list_collection_content(ResultType.document)).files
        for file in database_files:
            path = file.path
            if not os.path.isfile(path):
                orphanes.append(path)
                logger.debug(f"Discovered orphaned file: {path}")

        self.update_config(Config(rm_paths=orphanes))
        await self.delete()

        return len(orphanes)

    def get_embedding(self, texts: str | list[str]) -> list[NDArray]:
        """
        Generate embeddings and truncate them to `self._configs.embedding_dims` if needed.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = get_embedding_function(self._configs)(texts)
        if self._configs.embedding_dims:
            embeddings = [e[: self._configs.embedding_dims] for e in embeddings]
        return embeddings
