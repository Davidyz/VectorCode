import asyncio
from asyncio.subprocess import Process
import logging
import subprocess
import os
import socket
import sys
from typing import override


from vectorcode.cli_utils import Config, expand_path
from vectorcode.db.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


class LocalChromaVectorStore(ChromaVectorStore):
    """ChromaDB implementation of the vector store."""

    _process: Process | None = None
    _full_path: str

    def __init__(self, configs: Config):
        super().__init__(configs)
        self._full_path = str(expand_path(str(configs.project_root), absolute=True))

    async def _start_chroma_process(self) -> None:
        if self._process is not None:
            return

        assert self._configs.db_path is not None, "ChromaDB db_path must be set."
        db_path = os.path.expanduser(self._configs.db_path)
        self._configs.db_log_path = os.path.expanduser(self._configs.db_log_path)
        if not os.path.isdir(self._configs.db_log_path):
            os.makedirs(self._configs.db_log_path)
        if not os.path.isdir(db_path):
            logger.warning(
                f"Using local database at {os.path.expanduser('~/.local/share/vectorcode/chromadb/')}.",
            )
            db_path = os.path.expanduser("~/.local/share/vectorcode/chromadb/")
        env = os.environ.copy()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))  # OS selects a free ephemeral port
            self._chroma_settings.chroma_server_http_port = int(s.getsockname()[1])

        server_url = f"http://127.0.0.1:{self._chroma_settings.chroma_server_http_port}"
        logger.info(f"Starting bundled ChromaDB server at {server_url}.")
        env.update({"ANONYMIZED_TELEMETRY": "False"})

        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "chromadb.cli.cli",
            "run",
            "--host",
            "localhost",
            "--port",
            str(self._chroma_settings.chroma_server_http_port),
            "--path",
            db_path,
            "--log-path",
            os.path.join(str(self._configs.db_log_path), "chroma.log"),
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
            env=env,
        )

    @override
    async def connect(self) -> None:
        """Establish connection to ChromaDB."""
        if self._process is None:
            await self._start_chroma_process()
            # Wait for server to start up
            await asyncio.sleep(2)

        # we have to wait until the local chroma server is ready
        # Retry connection with exponential backoff
        max_retries = 5
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                await super().connect()
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.debug(
                    f"Connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    @override
    async def disconnect(self) -> None:
        """Close connection to ChromaDB."""
        if self._process is None:
            return

        logger.info("Shutting down the bundled Chromadb instance.")
        self._process.terminate()
        await self._process.wait()
