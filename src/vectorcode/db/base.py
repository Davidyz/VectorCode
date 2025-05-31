from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

from vectorcode.cli_utils import Config


class VectorStoreConnectionError(Exception):
    pass


class VectorStore(ABC):
    """Base class for vector database implementations.

    This abstract class defines the interface that all vector database implementations
    must follow. It provides methods for common vector database operations like
    querying, adding, and deleting vectors.
    """

    _configs: Config

    def __init__(self, configs: Config):
        self.__COLLECTION_CACHE: dict[str, Any] = {}
        self._configs = configs

        assert configs.project_root is not None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the vector database."""
        pass

    # @abstractmethod
    # async def check_health(self) -> None:
    #     """Check if the database is healthy and accessible. Raises a VectorStoreConnectionError if not."""
    #     pass

    @abstractmethod
    async def get_collection(
        self,
        collection_name: str,
        collection_meta: dict[str, Any] | None = None,
        make_if_missing: bool = False,
    ) -> Any:
        """Get an existing collection."""
        pass

    # @abstractmethod
    # async def get_or_create_collection(
    #     self,
    #     collection_name: str,
    #     metadata: Optional[Dict[str, Any]] = None,
    #     embedding_function: Optional[Any] = None,
    # ) -> Any:
    #     """Get an existing collection or create a new one if it doesn't exist."""
    #     pass

    # @abstractmethod
    # async def query(
    #     self,
    #     collection: Any,
    #     query_texts: List[str],
    #     n_results: int,
    #     where: Optional[Dict[str, Any]] = None,
    #     include: Optional[List[str]] = None,
    # ) -> Dict[str, Any]:
    #     """Query the vector database for similar vectors."""
    #     pass
    #
    # @abstractmethod
    # async def add(
    #     self,
    #     collection: Any,
    #     documents: List[str],
    #     metadatas: List[Dict[str, Any]],
    #     ids: Optional[List[str]] = None,
    # ) -> None:
    #     """Add documents to the vector database."""
    #     pass
    #
    # @abstractmethod
    # async def delete(
    #     self,
    #     collection: Any,
    #     where: Optional[Dict[str, Any]] = None,
    # ) -> None:
    #     """Delete documents from the vector database."""
    #     pass
    #
    # @abstractmethod
    # async def count(
    #     self,
    #     collection: Any,
    # ) -> int:
    #     """Get the number of documents in the collection."""
    #     pass
    #
    # @abstractmethod
    # async def get(
    #     self,
    #     collection: Any,
    #     ids: Union[str, List[str]],
    #     include: Optional[List[str]] = None,
    # ) -> Dict[str, Any]:
    #     """Get documents by their IDs."""
    #     pass

    def print_config(self) -> None:
        """Print the current database configuration."""
        parsed_url = urlparse(self._configs.db_url)

        print(f"{self._configs.db_type.title()} Configuration:")
        print(f"  URL: {self._configs.db_url}")
        print(f"  Host: {parsed_url.hostname or 'localhost'}")
        print(
            f"  Port: {parsed_url.port or (8000 if self._configs.db_type == 'chroma' else 6333)}"
        )
        print(f"  SSL: {parsed_url.scheme == 'https'}")
        if self._configs.db_settings:
            print("  Settings:")
            for key, value in self._configs.db_settings.items():
                print(f"    {key}: {value}")
