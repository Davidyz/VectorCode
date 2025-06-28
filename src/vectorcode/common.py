import asyncio
import contextlib
import hashlib
import logging
import os
import socket
import subprocess
import sys
from asyncio.subprocess import Process
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlparse

import chromadb
import httpx
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import IncludeEnum
from chromadb.config import APIVersion, Settings
from chromadb.utils import embedding_functions

from vectorcode.cli_utils import Config, LockManager, expand_path

logger = logging.getLogger(name=__name__)


async def get_collections(
    client: AsyncClientAPI,
) -> AsyncGenerator[AsyncCollection, None]:
    for collection_name in await client.list_collections():
        collection = await client.get_collection(collection_name, None)
        meta = collection.metadata
        if meta is None:
            continue
        if meta.get("created-by") != "VectorCode":
            continue
        if meta.get("username") not in (
            os.environ.get("USER"),
            os.environ.get("USERNAME"),
            "DEFAULT_USER",
        ):
            continue
        if meta.get("hostname") != socket.gethostname():
            continue
        yield collection


async def try_server(base_url: str):
    for ver in ("v1", "v2"):  # v1 for legacy, v2 for latest chromadb.
        heartbeat_url = f"{base_url}/api/{ver}/heartbeat"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url=heartbeat_url)
                logger.debug(f"Heartbeat {heartbeat_url} returned {response=}")
                if response.status_code == 200:
                    return True
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
    return False


async def wait_for_server(url: str, timeout=10):
    # Poll the server until it's ready or timeout is reached

    start_time = asyncio.get_event_loop().time()
    while True:
        if await try_server(url):
            return

        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(f"Server did not start within {timeout} seconds.")

        await asyncio.sleep(0.1)  # Wait before retrying


async def start_server(configs: Config):
    assert configs.db_path is not None
    db_path = os.path.expanduser(configs.db_path)
    configs.db_log_path = os.path.expanduser(configs.db_log_path)
    if not os.path.isdir(configs.db_log_path):
        os.makedirs(configs.db_log_path)
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
    process = await asyncio.create_subprocess_exec(
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
        os.path.join(str(configs.db_log_path), "chroma.log"),
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
        env=env,
    )

    await wait_for_server(server_url)
    configs.db_url = server_url
    return process


def get_collection_name(full_path: str) -> str:
    full_path = str(expand_path(full_path, absolute=True))
    hasher = hashlib.sha256()
    plain_collection_name = f"{os.environ.get('USER', os.environ.get('USERNAME', 'DEFAULT_USER'))}@{socket.gethostname()}:{full_path}"
    hasher.update(plain_collection_name.encode())
    collection_id = hasher.hexdigest()[:63]
    logger.debug(
        f"Hashing {plain_collection_name} as the collection name for {full_path}."
    )
    return collection_id


def get_embedding_function(configs: Config) -> chromadb.EmbeddingFunction | None:
    try:
        return getattr(embedding_functions, configs.embedding_function)(
            **configs.embedding_params
        )
    except AttributeError:
        logger.warning(
            f"Failed to use {configs.embedding_function}. Falling back to Sentence Transformer.",
        )
        return embedding_functions.SentenceTransformerEmbeddingFunction()  # type:ignore
    except Exception as e:
        e.add_note(
            "\nFor errors caused by missing dependency, consult the documentation of pipx (or whatever package manager that you installed VectorCode with) for instructions to inject libraries into the virtual environment."
        )
        logger.error(
            f"Failed to use {configs.embedding_function} with following error.",
        )
        raise


__COLLECTION_CACHE: dict[str, AsyncCollection] = {}


async def get_collection(
    client: AsyncClientAPI, configs: Config, make_if_missing: bool = False
):
    """
    Raise ValueError when make_if_missing is False and no collection is found;
    Raise IndexError on hash collision.
    """
    assert configs.project_root is not None
    full_path = str(expand_path(str(configs.project_root), absolute=True))
    if __COLLECTION_CACHE.get(full_path) is None:
        collection_name = get_collection_name(full_path)
        embedding_function = get_embedding_function(configs)

        collection_meta: dict[str, str | int] = {
            "path": full_path,
            "hostname": socket.gethostname(),
            "created-by": "VectorCode",
            "username": os.environ.get(
                "USER", os.environ.get("USERNAME", "DEFAULT_USER")
            ),
            "embedding_function": configs.embedding_function,
            "hnsw:M": 64,
        }
        if configs.hnsw:
            for key in configs.hnsw.keys():
                target_key = key
                if not key.startswith("hnsw:"):
                    target_key = f"hnsw:{key}"
                collection_meta[target_key] = configs.hnsw[key]
        logger.debug(
            f"Getting/Creating collection with the following metadata: {collection_meta}"
        )
        if not make_if_missing:
            __COLLECTION_CACHE[full_path] = await client.get_collection(
                collection_name, embedding_function
            )
        else:
            collection = await client.get_or_create_collection(
                collection_name,
                metadata=collection_meta,
                embedding_function=embedding_function,
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
                    f"Failed to use existing collection due to metadata mismatch: {collection_meta}"
                )
                raise IndexError(
                    "Failed to create the collection due to hash collision. Please file a bug report."
                )
            __COLLECTION_CACHE[full_path] = collection
    return __COLLECTION_CACHE[full_path]


def verify_ef(collection: AsyncCollection, configs: Config):
    collection_ef = collection.metadata.get("embedding_function")
    collection_ep = collection.metadata.get("embedding_params")
    if collection_ef and collection_ef != configs.embedding_function:
        logger.error(f"The collection was embedded using {collection_ef}.")
        logger.error(
            "Embeddings and query must use the same embedding function and parameters. Please double-check your config."
        )
        return False
    elif collection_ep and collection_ep != configs.embedding_params:
        logger.warning(
            f"The collection was embedded with a different set of configurations: {collection_ep}. The result may be inaccurate.",
        )
    return True


async def list_collection_files(collection: AsyncCollection) -> list[str]:
    return list(
        set(
            str(c.get("path", None))
            for c in (await collection.get(include=[IncludeEnum.metadatas])).get(
                "metadatas"
            )
            or []
        )
    )


@dataclass
class _ClientModel:
    client: AsyncClientAPI
    is_bundled: bool = False
    process: Optional[Process] = None


class ClientManager:
    singleton: Optional["ClientManager"] = None
    __clients: dict[str, _ClientModel]

    def __new__(cls) -> "ClientManager":
        if cls.singleton is None:
            cls.singleton = super().__new__(cls)
            cls.singleton.__clients = {}
        return cls.singleton

    @contextlib.asynccontextmanager
    async def get_client(self, configs: Config, need_lock: bool = True):
        project_root = str(expand_path(str(configs.project_root), True))
        is_bundled = False
        if self.__clients.get(project_root) is None:
            process = None
            if not await try_server(configs.db_url):
                logger.info(f"Starting a new server at {configs.db_url}")
                process = await start_server(configs)
                is_bundled = True

            self.__clients[project_root] = _ClientModel(
                client=await self._create_client(configs),
                is_bundled=is_bundled,
                process=process,
            )
        lock = None
        if self.__clients[project_root].is_bundled and need_lock:
            lock = LockManager().get_lock(str(configs.db_path))
            logger.debug(f"Locking {configs.db_path}")
            await lock.acquire()
        yield self.__clients[project_root].client
        if lock is not None:
            logger.debug(f"Unlocking {configs.db_path}")
            await lock.release()

    def get_processes(self) -> list[Process]:
        return [i.process for i in self.__clients.values() if i.process is not None]

    async def kill_servers(self):
        termination_tasks: list[asyncio.Task] = []
        for p in self.get_processes():
            logger.info(f"Killing bundled chroma server with PID: {p.pid}")
            p.terminate()
            termination_tasks.append(asyncio.create_task(p.wait()))
        await asyncio.gather(*termination_tasks)

    async def _create_client(self, configs: Config) -> AsyncClientAPI:
        settings: dict[str, Any] = {"anonymized_telemetry": False}
        if isinstance(configs.db_settings, dict):
            valid_settings = {
                k: v for k, v in configs.db_settings.items() if k in Settings.__fields__
            }
            settings.update(valid_settings)
        parsed_url = urlparse(configs.db_url)
        settings["chroma_server_host"] = parsed_url.hostname or "127.0.0.1"
        settings["chroma_server_http_port"] = parsed_url.port or 8000
        settings["chroma_server_ssl_enabled"] = parsed_url.scheme == "https"
        settings["chroma_server_api_default_path"] = parsed_url.path or APIVersion.V2
        settings_obj = Settings(**settings)
        return await chromadb.AsyncHttpClient(
            settings=settings_obj,
            host=str(settings_obj.chroma_server_host),
            port=int(settings_obj.chroma_server_http_port or 8000),
        )

    def clear(self):
        self.__clients.clear()
