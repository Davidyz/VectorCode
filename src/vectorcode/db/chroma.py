import logging
import os
import socket
from typing import Any, Dict, override

import chromadb
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.config import Settings

from vectorcode.cli_utils import Config
from vectorcode.db.base import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the vector store."""

    _client: AsyncClientAPI | None = None
    _chroma_settings: Settings

    def __init__(self, configs: Config):
        super().__init__(configs)
        settings: Dict[str, Any] = {"anonymized_telemetry": False}
        if isinstance(self.configs.db_settings, dict):
            valid_settings = {
                k: v
                for k, v in self.configs.db_settings.items()
                if k in Settings.__fields__
            }
            settings.update(valid_settings)

        from urllib.parse import urlparse

        parsed_url = urlparse(self.configs.db_url)
        settings["chroma_server_host"] = parsed_url.hostname or "127.0.0.1"
        settings["chroma_server_http_port"] = parsed_url.port or 8000
        settings["chroma_server_ssl_enabled"] = parsed_url.scheme == "https"
        settings["chroma_server_api_default_path"] = "/api/v2"

        self._chroma_settings = Settings(**settings)

    async def connect(self) -> None:
        """Establish connection to ChromaDB."""
        try:
            self._client = await chromadb.AsyncHttpClient(
                settings=self._chroma_settings,
                host=str(self._chroma_settings.chroma_server_host),
                port=int(self._chroma_settings.chroma_server_http_port or 8000),
            )
            await self.check_health()
        except Exception as e:
            logger.error(f"Could not connect to ChromaDB: {e}")

    @override
    async def check_health(self) -> bool:
        try:
            if self._client is None:
                await self.connect()

            assert self._client is not None, "Chroma client is not connected."
            await self._client.heartbeat()

            return True
        except Exception as e:
            logger.error(f"ChromaDB is not healthy: {e}")
            return False

    async def disconnect(self) -> None:
        """Close connection to ChromaDB."""
        return None

    async def get_collection(
        self,
        make_if_missing: bool = False,
    ) -> AsyncCollection:
        """
        Raise ValueError when make_if_missing is False and no collection is found;
        Raise IndexError on hash collision.
        """
        if not self._client:
            await self.connect()

        assert self._client is not None, "Chroma client is not connected."

        if self.__COLLECTION_CACHE.get(self.full_path) is None:
            if not make_if_missing:
                self.__COLLECTION_CACHE[
                    self.full_path
                ] = await self._client.get_collection(
                    self.collection_name, self.embedding_function
                )
            else:
                collection = await self._client.get_or_create_collection(
                    self.collection_name,
                    metadata=self.collection_metadata,
                    embedding_function=self.embedding_function,
                )
                if (
                    not collection.metadata.get("hostname") == socket.gethostname()
                    or collection.metadata.get("username")
                    not in (
                        os.environ.get("USER"),
                        os.environ.get("USERNAME"),
                        "DEFAULT_USER",
                    )
                    or not collection.metadata.get("created-by") == "VectorCode"
                ):
                    logger.error(
                        f"Failed to use existing collection due to metadata mismatch: {self.collection_metadata}"
                    )
                    raise IndexError(
                        "Failed to create the collection due to hash collision. Please file a bug report."
                    )
                self.__COLLECTION_CACHE[self.full_path] = collection
        return self.__COLLECTION_CACHE[self.full_path]
