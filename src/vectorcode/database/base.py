import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Self, Sequence

from numpy.typing import NDArray

from vectorcode.chunking import Chunk, TreeSitterChunker
from vectorcode.cli_utils import Config
from vectorcode.database.types import (
    CollectionContent,
    CollectionInfo,
    QueryResult,
    ResultType,
    VectoriseStats,
)
from vectorcode.database.utils import get_embedding_function

logger = logging.getLogger(name=__name__)


"""
For developers:

To implement a custom database connector, you should inherit the following 
`DatabaseConnectorBase` class and implement ALL abstract methods. 

You should also try to wrap the exceptions with the ones in 
`src/vectorcode/database/errors.py` where appropriate, because this helps the 
CLI/LSP/MCP interfaces to handle some common edge cases (for example, querying 
from an unindexed project). To do this, you should do the following in a 
try-except block:
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
        """
        Create a new instance of the database connector.
        This classmethod will add the docstring of the child class to the exception if the initialisation fails.
        """
        try:
            return cls(configs)
        except Exception as e:  # pragma: nocover
            doc_string = cls.__doc__
            if doc_string:
                e.add_note(doc_string)
            raise

    def __init__(self, configs: Config):
        """
        Initialises the database connector with the given configs.
        It is recommended to use the `create` classmethod instead of calling this directly,
        as it provides better error handling during initialisation.
        """
        self._configs = configs

    async def count(self, what: ResultType = ResultType.chunk) -> int:
        """
        Returns the chunk count or file count of the given collection, depending on the value passed for `what`.
        This method is implemented in the base class and relies on `list_collection_content`.
        Child classes should not need to override this method if `list_collection_content` is implemented correctly.
        """
        collection_content = await self.list_collection_content(what=what)
        match what:
            case ResultType.chunk:
                return len(collection_content.chunks)
            case ResultType.document:
                return len(collection_content.files)

    @abstractmethod
    async def query(
        self,
    ) -> list[QueryResult]:
        """
        Query the database for similar chunks.
        The query keywords are stored in `self._configs.query`.
        The implementation of this method should handle the conversion from the native database query result to a list of `vectorcode.database.types.QueryResult` objects.
        """
        pass

    @abstractmethod
    async def vectorise(
        self,
        file_path: str,
        chunker: TreeSitterChunker | None = None,
    ) -> VectoriseStats:
        """
        Vectorise the given file and add it to the database.
        The duplicate checking (using file hash) should be done outside of this function.

        For developers:
        The implementation should chunk the file, generate embeddings for the chunks, and store them in the database.
        It should return a `VectoriseStats` object to report the outcome.
        """
        pass

    @abstractmethod
    async def list_collections(self) -> Sequence[CollectionInfo]:
        """
        List all collections available in the database.

        For developers:
        The implementation should retrieve all collections and return them as a sequence of `CollectionInfo` objects.
        This includes metadata about each collection like its ID, path, and size.
        """
        pass

    @abstractmethod
    async def list_collection_content(
        self,
        *,
        what: Optional[ResultType] = None,
        collection_id: str | None = None,
        collection_path: str | None = None,
    ) -> CollectionContent:
        """
        List the content of a collection (from `self._configs.project_root`).
        You may use the keyword arguments to temporarily override the collection of interests.

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

        For developers:
        The file paths to be deleted are stored in `self._configs.rm_paths`.
        The implementation should remove all chunks associated with these files from the database.
        """
        pass

    @abstractmethod
    async def drop(
        self, *, collection_id: str | None = None, collection_path: str | None = None
    ):
        """
        Delete a collection from the database.
        The collection to be dropped is specified by `collection_id` or `collection_path`.
        If not provided, it defaults to `self._configs.project_root`.
        """
        pass

    def _check_new_config(self, new_config: Config) -> bool:
        """
        Ensures that the new config does not attempt to change database-specific settings.
        It copies the `db_type` and `db_params` from the existing config to the new one.
        This is a helper method for `update_config` and `replace_config`.
        """
        assert isinstance(new_config, Config), "`new_config` is not a `Config` object."
        new_config.db_type = self._configs.db_type
        new_config.db_params = self._configs.db_params
        return True

    async def update_config(self, new_config: Config) -> Self:
        """
        Merge the new config with the existing one.
        This method will not change the database configs.
        Child classes should not need to override this method.
        """
        assert self._check_new_config(new_config), (
            "The new config has different database configs."
        )

        self._configs = await self._configs.merge_from(new_config)

        return self

    async def replace_config(self, new_config: Config) -> Self:
        """
        Replace the existing config with the new one.
        This method will not change the database configs.
        Child classes should not need to override this method.
        """
        assert self._check_new_config(new_config), (
            "The new config has different database configs."
        )
        self._configs = new_config
        return self

    async def check_orphanes(self) -> int:
        """
        Check for files that are in the database but no longer on disk, and remove them.
        Returns the number of orphaned files removed.
        This method relies on `list_collection_content` and `delete`.
        Child classes should not need to override this.
        """

        orphanes: list[str] = []
        database_files = (
            await self.list_collection_content(what=ResultType.document)
        ).files
        for file in database_files:
            path = file.path
            if not os.path.isfile(path):
                orphanes.append(path)
                logger.debug(f"Discovered orphaned file: {path}")

        await self.update_config(Config(rm_paths=orphanes))
        await self.delete()

        return len(orphanes)

    def get_embedding(self, texts: str | list[str]) -> list[NDArray]:
        """
        Generate embeddings for the given texts.
        It uses the embedding function specified in `self._configs.embedding_function`.
        If `self._configs.embedding_dims` is set, it truncates the embeddings.
        Child classes should use this method to get embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return []
        texts = [i for i in texts]
        logger.debug(f"Getting embeddings for {texts}")
        embeddings = get_embedding_function(self._configs)(texts)
        if self._configs.embedding_dims:
            embeddings = [e[: self._configs.embedding_dims] for e in embeddings]
        return embeddings

    @abstractmethod
    async def get_chunks(self, file_path) -> list[Chunk]:
        """
        Retrieve all chunks for a given file from the database.
        If the file is not found in the database, it should return an empty list.

        For developers:
        This is useful for operations that need to inspect the chunked content of a file, for example, for debugging or analysis.
        """
        pass

    async def cleanup(self) -> list[str]:
        """
        Remove empty collections from the database.
        Returns a list of paths of the removed collections.
        This method relies on `list_collections` and `drop`.
        Child classes should not need to override this.
        """
        removed: list[str] = []
        for collection in await self.list_collections():
            if collection.chunk_count == 0:
                removed.append(collection.path)
                await self.drop(collection_path=collection.path)

        return removed
