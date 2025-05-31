import logging
from typing import Any, Dict, override
from urllib.parse import urlparse

import chromadb
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from vectorcode.cli_utils import Config
from vectorcode.db.base import VectorStore, VectorStoreConnectionError

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the vector store."""

    _client: AsyncClientAPI | None = None
    _chroma_settings: Settings
    _embedding_function: chromadb.EmbeddingFunction | None

    def __init__(self, configs: Config):
        super().__init__(configs)
        settings: Dict[str, Any] = {"anonymized_telemetry": False}
        if isinstance(self._configs.db_settings, dict):
            valid_settings = {
                k: v
                for k, v in self._configs.db_settings.items()
                if k in Settings.__fields__
            }
            settings.update(valid_settings)

        parsed_url = urlparse(self._configs.db_url)
        settings["chroma_server_host"] = parsed_url.hostname or "127.0.0.1"
        settings["chroma_server_http_port"] = parsed_url.port or 8000
        settings["chroma_server_ssl_enabled"] = parsed_url.scheme == "https"
        settings["chroma_server_api_default_path"] = "/api/v2"

        self._chroma_settings = Settings(**settings)

        try:
            self._embedding_function = getattr(
                embedding_functions, configs.embedding_function
            )(**configs.embedding_params)
        except AttributeError:
            logger.warning(
                f"Failed to use {configs.embedding_function}. Falling back to Sentence Transformer.",
            )
            self._embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction()  # type:ignore
            )
        except Exception as e:
            e.add_note(
                "\nFor errors caused by missing dependency, consult the documentation of pipx (or whatever package manager that you installed VectorCode with) for instructions to inject libraries into the virtual environment."
            )
            logger.error(
                f"Failed to use {configs.embedding_function} with following error.",
            )
            raise

    @override
    async def connect(self) -> None:
        """Establish connection to ChromaDB."""
        try:
            if self._client is None:
                logger.debug(
                    f"Connecting to ChromaDB at {self._chroma_settings.chroma_server_host}:{self._chroma_settings.chroma_server_http_port}."
                )
                self._client = await chromadb.AsyncHttpClient(
                    settings=self._chroma_settings,
                    host=str(self._chroma_settings.chroma_server_host),
                    port=int(self._chroma_settings.chroma_server_http_port or 8000),
                )

            await self._client.heartbeat()
        except Exception as e:
            logger.error(f"Could not connect to ChromaDB: {e}")
            raise VectorStoreConnectionError(e)

    @override
    async def disconnect(self) -> None:
        """Not required for non local chromadb."""
        pass

    @override
    async def get_collection(
        self,
        collection_name: str,
        collection_meta: dict[str, Any] | None = None,
        make_if_missing: bool = False,
    ) -> AsyncCollection:
        """
        Raise ValueError when make_if_missing is False and no collection is found;
        Raise IndexError on hash collision.
        """
        if not self._client:
            await self.connect()

        assert self._client is not None, "Chroma client is not connected."

        if not make_if_missing:
            return await self._client.get_collection(collection_name)
        else:
            return await self._client.get_or_create_collection(
                collection_name,
                metadata=collection_meta,
                embedding_function=self._embedding_function,
            )
