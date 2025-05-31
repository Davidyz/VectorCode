import asyncio
from asyncio.subprocess import Process
import logging
import subprocess
import os
import socket
import sys
from typing import Any, Dict, override

import chromadb
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.config import Settings

from vectorcode.cli_utils import Config
from vectorcode.db.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


class LocalChromaVectorStore(ChromaVectorStore):
    """ChromaDB implementation of the vector store."""

    _client: AsyncClientAPI | None = None
    _process: Process | None = None
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

    async def _start_chroma_process(self) -> None:
        if self._process is not None:
            return

        assert self.configs.db_path is not None, "ChromaDB db_path must be set."
        db_path = os.path.expanduser(self.configs.db_path)
        self.configs.db_log_path = os.path.expanduser(self.configs.db_log_path)
        if not os.path.isdir(self.configs.db_log_path):
            os.makedirs(self.configs.db_log_path)
        if not os.path.isdir(db_path):
            logger.warning(
                f"Using local database at {os.path.expanduser('~/.local/share/vectorcode/chromadb/')}.",
            )
            db_path = os.path.expanduser("~/.local/share/vectorcode/chromadb/")
        env = os.environ.copy()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))  # OS selects a free ephemeral port
            port = int(s.getsockname()[1])

        server_url = f"http://127.0.0.1:{port}"
        logger.warning(f"Starting bundled ChromaDB server at {server_url}.")
        env.update({"ANONYMIZED_TELEMETRY": "False"})

        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "chromadb.cli.cli",
            "run",
            "--host",
            "localhost",
            "--port",
            str(port),
            "--path",
            db_path,
            "--log-path",
            os.path.join(str(self.configs.db_log_path), "chroma.log"),
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
            env=env,
        )

    async def connect(self) -> None:
        """Establish connection to ChromaDB."""
        if self._process is None:
            await self._start_chroma_process()

        try:
            self._client = await chromadb.AsyncHttpClient(
                settings=self._chroma_settings,
                host=str(self._chroma_settings.chroma_server_host),
                port=int(self._chroma_settings.chroma_server_http_port or 8000),
            )
            await self.check_health()
        except Exception as e:
            logger.error(f"Could not connect to ChromaDB: {e}")

    # @override
    # async def check_health(self) -> bool:
    #     try:
    #         if self._client is None:
    #             await self.connect()
    #
    #         assert self._client is not None, "Chroma client is not connected."
    #         await self._client.heartbeat()
    #
    #         return True
    #     except Exception as e:
    #         logger.error(f"ChromaDB is not healthy: {e}")
    #         return False

    async def disconnect(self) -> None:
        """Close connection to ChromaDB."""
        if self._process is None:
            return

        logger.info("Shutting down the bundled Chromadb instance.")
        self._process.terminate()
        await self._process.wait()

    # async def get_collection(
    #     self,
    #     make_if_missing: bool = False,
    # ) -> AsyncCollection:
    #     """
    #     Raise ValueError when make_if_missing is False and no collection is found;
    #     Raise IndexError on hash collision.
    #     """
    #     if not self._client:
    #         await self.connect()
    #
    #     assert self._client is not None, "Chroma client is not connected."
    #
    #     if self.__COLLECTION_CACHE.get(self.full_path) is None:
    #         if not make_if_missing:
    #             self.__COLLECTION_CACHE[
    #                 self.full_path
    #             ] = await self._client.get_collection(
    #                 self.collection_name, self.embedding_function
    #             )
    #         else:
    #             collection = await self._client.get_or_create_collection(
    #                 self.collection_name,
    #                 metadata=self.collection_metadata,
    #                 embedding_function=self.embedding_function,
    #             )
    #             if (
    #                 not collection.metadata.get("hostname") == socket.gethostname()
    #                 or collection.metadata.get("username")
    #                 not in (
    #                     os.environ.get("USER"),
    #                     os.environ.get("USERNAME"),
    #                     "DEFAULT_USER",
    #                 )
    #                 or not collection.metadata.get("created-by") == "VectorCode"
    #             ):
    #                 logger.error(
    #                     f"Failed to use existing collection due to metadata mismatch: {self.collection_metadata}"
    #                 )
    #                 raise IndexError(
    #                     "Failed to create the collection due to hash collision. Please file a bug report."
    #                 )
    #             self.__COLLECTION_CACHE[self.full_path] = collection
    #     return self.__COLLECTION_CACHE[self.full_path]
