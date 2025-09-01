import asyncio
import contextlib
import copy
import logging
import os
import socket
import subprocess
import sys
from asyncio.subprocess import Process
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import chromadb
import httpx
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import IncludeEnum, QueryResult
from chromadb.config import APIVersion, Settings
from tree_sitter import Point

import vectorcode.subcommands.query.types as vectorcode_query_types
from vectorcode.chunking import Chunk
from vectorcode.cli_utils import Config, LockManager, expand_path
from vectorcode.database.base import DatabaseConnectorBase
from vectorcode.database.types import (
    CollectionContent,
    CollectionInfo,
    FileInCollection,
    ResultType,
    VectoriseStats,
)
from vectorcode.subcommands.vectorise import get_uuid

logger = logging.getLogger(name=__name__)


def __convert_chroma_query_results(
    chroma_result: QueryResult, queries: list[str]
) -> list[vectorcode_query_types.QueryResult]:
    """Convert chromadb query result to in-house query results"""
    assert chroma_result["documents"] is not None
    assert chroma_result["distances"] is not None
    assert chroma_result["metadatas"] is not None
    assert chroma_result["ids"] is not None

    chroma_results_list: list[vectorcode_query_types.QueryResult] = []
    for q_i in range(len(queries)):
        q = queries[q_i]
        documents = chroma_result["documents"][q_i]
        distances = chroma_result["distances"][q_i]
        metadatas = chroma_result["metadatas"][q_i]
        ids = chroma_result["ids"][q_i]
        for doc, dist, meta, _id in zip(documents, distances, metadatas, ids):
            chunk = Chunk(text=doc, id=_id)
            if meta.get("start"):
                chunk.start = Point(int(meta.get("start", 0)), 0)
            if meta.get("end"):
                chunk.end = Point(int(meta.get("end", 0)), 0)
            if meta.get("path"):
                chunk.path = str(meta["path"])
            chroma_results_list.append(
                vectorcode_query_types.QueryResult(
                    chunk=chunk,
                    path=str(meta.get("path", "")),
                    query=(q,),
                    scores=(-dist,),
                )
            )
    return chroma_results_list


async def _try_server(base_url: str):
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


async def _wait_for_server(base_url: str, timeout: int = 10):
    # Poll the server until it's ready or timeout is reached

    start_time = asyncio.get_event_loop().time()
    while True:
        if await _try_server(base_url):
            return

        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(f"Server did not start within {timeout} seconds.")

        await asyncio.sleep(0.1)  # Wait before retrying


async def _start_server(configs: Config):
    assert configs.db_params.get("db_url") is not None
    db_path = os.path.expanduser(configs.db_params["db_path"])
    db_log_path = configs.db_params["db_log_path"]
    if not os.path.isdir(db_log_path):
        os.makedirs(db_log_path)
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
        os.path.join(str(db_log_path), "chroma.log"),
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
        env=env,
    )

    await _wait_for_server(server_url)
    configs.db_params["db_url"] = server_url
    return process


@dataclass
class _Chroma0ClientModel:
    client: AsyncClientAPI
    is_bundled: bool = False
    process: Optional[Process] = None


class Chroma0ClientManager:
    singleton: Optional["Chroma0ClientManager"] = None
    __clients: dict[str, _Chroma0ClientModel]

    def __new__(cls) -> "Chroma0ClientManager":
        if cls.singleton is None:
            cls.singleton = super().__new__(cls)
            cls.singleton.__clients = {}
        return cls.singleton

    @contextlib.asynccontextmanager
    async def get_client(self, configs: Config, need_lock: bool = True):
        project_root = str(expand_path(str(configs.project_root), True))
        is_bundled = False
        url = configs.db_params["db_url"]
        db_path = configs.db_params["db_path"]
        db_log_path = configs.db_params["db_log_path"]
        if self.__clients.get(project_root) is None:
            process = None
            if not await _try_server(url):
                logger.info(f"Starting a new server at {url}")
                process = await _start_server(url)
                is_bundled = True

            self.__clients[project_root] = _Chroma0ClientModel(
                client=await self._create_client(configs),
                is_bundled=is_bundled,
                process=process,
            )
        lock = None
        if self.__clients[project_root].is_bundled and need_lock:
            lock = LockManager().get_lock(str(db_path))
            logger.debug(f"Locking {db_path}")
            await lock.acquire()
        yield self.__clients[project_root].client
        if lock is not None:
            logger.debug(f"Unlocking {db_log_path}")
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
        db_settings = configs.db_params["db_settings"]
        if isinstance(db_settings, dict):
            valid_settings = {
                k: v for k, v in db_settings.items() if k in Settings.__fields__
            }
            settings.update(valid_settings)
        parsed_url = urlparse(configs.db_params["db_url"])
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


__default_settings: dict[str, Any] = {
    "db_url": "http://127.0.0.1",
    "db_path": os.path.expanduser("~/.local/share/vectorcode/chromadb/"),
    "db_log_path": os.path.expanduser("~/.local/share/vectorcode/"),
    "db_settings": {},
    "hnsw": {"hnsw:M": 64},
}


class ChromaDB0Connector(DatabaseConnectorBase):
    """
    This is the connector layer for **ChromaDB 0.6.3**

    Valid `db_params` options for ChromaDB 0.6.x:
        - `db_url`: default to `http://127.0.0.1:8000`
        - `db_path`: default to `~/.local/share/vectorcode/chromadb/`;
        - `db_log_path`: default to `~/.local/share/vectorcode/`
        - `db_settings`: See https://github.com/chroma-core/chroma/blob/a3b86a0302a385350a8f092a5f89a2dcdebcf6be/chromadb/config.py#L101
        - `hnsw`: default to `{ "hnsw:M": 64 }`
    """

    def __init__(self, configs: Config):
        super().__init__(configs)
        params = copy.deepcopy(__default_settings)
        params.update(self._configs.db_params)
        self._configs.db_params = params

    async def query(self, collection_id, keywords_embeddings, opts):
        assert len(opts.keywords), "Keywords cannot be empty"
        assert len(keywords_embeddings) == len(opts.keywords), (
            "Number of embeddings must match number of keywords."
        )
        async with Chroma0ClientManager().get_client(self._configs, False) as client:
            collection = await client.get_collection(collection_id)
            query_count = opts.count or (
                await self.count(collection_id, ResultType.chunk)
            )
            query_result = await collection.query(
                query_embeddings=keywords_embeddings,
                include=[
                    IncludeEnum.metadatas,
                    IncludeEnum.documents,
                    IncludeEnum.distances,
                ],
                n_results=query_count,
            )
            return __convert_chroma_query_results(query_result, opts.keywords)

    async def _create_or_get_collection(self, collection_id) -> AsyncCollection:
        """
        This method should be used by ChromaDB methods that are expected to **create a collection when not found**.
        For other methods, just use `client.get_collection` and let it fail if the collection doesn't exist.
        """
        assert self._configs.project_root is not None
        collection_meta: dict[str, str | int] = {
            "path": os.path.abspath(str(self._configs.project_root)),
            "hostname": socket.gethostname(),
            "created-by": "VectorCode",
            "username": os.environ.get(
                "USER", os.environ.get("USERNAME", "DEFAULT_USER")
            ),
            "embedding_function": self._configs.embedding_function,
        }
        db_params = self._configs.db_params
        for key in db_params.get("hnsw", {}).keys():
            meta_field_name: str = key
            if not meta_field_name.startswith("hnsw:"):
                meta_field_name = f"hnsw:{meta_field_name}"
            collection_meta[meta_field_name] = db_params[key]

        async with Chroma0ClientManager().get_client(self._configs, True) as client:
            col = await client.get_or_create_collection(
                collection_id, metadata=collection_meta
            )
            for key in collection_meta.keys():
                # validate metadata
                assert collection_meta[key] == col.metadata.get(key), (
                    f"Metadata field {key} mismatch!"
                )

            return col

    async def vectorise(self, collection_id, chunks, chunk_embeddings, file_hashes):
        # WIP: finish the stats.
        # should this method handle chunking and hash checking?
        stats = VectoriseStats()

        async with Chroma0ClientManager().get_client(self._configs, True) as client:
            collection = await self._create_or_get_collection(collection_id)
            max_batch_size = await client.get_max_batch_size()
            for idx_batch in range(0, len(chunks), max_batch_size):
                this_batch: list[Chunk] = chunks[idx_batch : idx_batch + max_batch_size]
                metadatas = []
                for idx in range(len(this_batch)):
                    this_chunk = this_batch[idx]
                    meta: dict[str, str | int] = {
                        "path": str(this_chunk.path),
                    }
                    if file_hashes:
                        for idx in range(idx_batch, idx_batch + max_batch_size):
                            meta["sha256"] = file_hashes[idx]
                    if this_chunk.start and isinstance(this_chunk.start.row, int):
                        meta["start"] = this_chunk.start.row
                    if this_chunk.end and isinstance(this_chunk.end.row, int):
                        meta["end"] = this_chunk.end.row

                    metadatas.append(meta)

                await collection.add(
                    documents=[chunk.text for chunk in this_batch],
                    ids=[get_uuid() for _ in range(max_batch_size)],
                    embeddings=chunk_embeddings[idx_batch : idx_batch + max_batch_size],
                    metadatas=metadatas,
                )
        return stats

    async def list_collections(self):
        async with Chroma0ClientManager().get_client(
            self._configs, need_lock=False
        ) as client:
            result: list[CollectionInfo] = []
            for col_name in await client.list_collections():
                col = await client.get_collection(col_name)
                col_counts = await self.list(col_name)
                result.append(
                    CollectionInfo(
                        id=col_name,
                        path=str(col.metadata.get("path")),
                        embedding_function=col.metadata.get(
                            "embedding_function",
                            Config().embedding_function,  # fallback to default
                        ),
                        database_backend="Chroma0",
                        file_count=len(col_counts.files),
                        chunk_count=len(col_counts.chunks),
                    )
                )
        return result

    async def list(self, collection_id, what=None) -> CollectionContent:
        """
        When `what` is None, this method should populate both `CollectionContent.files` and `CollectionContent.chunks`.
        Otherwise, this method may populate only one of them to save waiting time.
        """
        content = CollectionContent()
        async with Chroma0ClientManager().get_client(
            configs=self._configs, need_lock=False
        ) as client:
            collection = await client.get_collection(collection_id)
            raw_content = await collection.get(
                include=[
                    IncludeEnum.metadatas,
                    IncludeEnum.documents,
                ]
            )
            metadatas = raw_content.get("metadatas", [])
            documents = raw_content.get("documents", [])
            ids = raw_content.get("ids", [])
            assert metadatas
            assert documents
            assert ids
            if what is None or what == ResultType.document:
                content.files.extend(
                    set(
                        FileInCollection(
                            path=str(i.get("path")), sha256=str(i.get("sha256"))
                        )
                        for i in metadatas
                    )
                )
            if what is None or what == ResultType.chunk:
                for i in range(len(ids)):
                    start, end = None, None
                    if metadatas[i].get("start") is not None:
                        start = Point(row=int(metadatas[i]["start"]), column=0)
                    if metadatas[i].get("end") is not None:
                        end = Point(row=int(metadatas[i]["end"]), column=0)
                    content.chunks.append(
                        Chunk(
                            text=documents[i],
                            path=str(metadatas[i].get("path", "")) or None,
                            id=ids[i],
                            start=start,
                            end=end,
                        )
                    )

        return content
