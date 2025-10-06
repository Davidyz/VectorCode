import asyncio
import contextlib
import logging
import os
import socket
import sys
from typing import Any, Literal, Optional, Sequence, cast
from urllib.parse import urlparse

import chromadb
from filelock import AsyncFileLock
from tree_sitter import Point

from vectorcode.chunking import Chunk, TreeSitterChunker
from vectorcode.cli_utils import (
    Config,
    LockManager,
    QueryInclude,
    expand_globs,
    expand_path,
)
from vectorcode.database import DatabaseConnectorBase
from vectorcode.database.chroma_common import convert_chroma_query_results
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.database.types import (
    CollectionContent,
    CollectionInfo,
    FileInCollection,
    QueryResult,
    ResultType,
    VectoriseStats,
)
from vectorcode.database.utils import get_collection_id, get_uuid, hash_file

if not chromadb.__version__.startswith("1."):
    logging.error(
        f"""
Found ChromaDB {chromadb.__version__}, which is incompatible wiht your VectorCode installation. Please install `vectorcode`.

For example:
uv tool install vectorcode
"""
    )
    sys.exit(1)


from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.config import APIVersion, Settings
from chromadb.errors import NotFoundError

logger = logging.getLogger(name=__name__)

SupportedClientType = Literal["http"] | Literal["persistent"]

_SUPPORTED_CLIENT_TYPE: set[SupportedClientType] = {"http", "persistent"}

_default_settings: dict[str, Any] = {
    "db_url": None,
    "db_path": os.path.expanduser("~/.local/share/vectorcode/chromadb/"),
    "db_log_path": os.path.expanduser("~/.local/share/vectorcode/"),
    "db_settings": {},
    "hnsw": {"hnsw:M": 64},
}


class ChromaDBConnector(DatabaseConnectorBase):
    """
    This is the connector layer for **ChromaDB 1.x**

    Valid `db_params` options for ChromaDB 1.x:
        - `db_url`: default to `http://127.0.0.1:8000`
        - `db_path`: default to `~/.local/share/vectorcode/chromadb/`;
        - `db_log_path`: default to `~/.local/share/vectorcode/`
        - `db_settings`: See https://github.com/chroma-core/chroma/blob/508080841d2b2ebb3a9fbdc612087248df6f1382/chromadb/config.py#L120
        - `hnsw`: default to `{ "hnsw:M": 64 }`
    """

    def __init__(self, configs: Config):
        super().__init__(configs)
        params = _default_settings.copy()
        params.update(self._configs.db_params.copy())
        params["db_path"] = os.path.expanduser(params["db_path"])
        params["db_log_path"] = os.path.expanduser(params["db_log_path"])
        self._configs.db_params = params

        self._lock: AsyncFileLock | None = None
        self._client: ClientAPI | None = None
        self._client_type: SupportedClientType

    def _create_client(self) -> ClientAPI:
        global _SUPPORTED_CLIENT_TYPE
        settings: dict[str, Any] = {"anonymized_telemetry": False}
        db_params = self._configs.db_params
        settings.update(db_params["db_settings"])
        if db_params.get("db_url"):
            parsed_url = urlparse(db_params["db_url"])

            settings["chroma_server_host"] = settings.get(
                "chroma_server_host", parsed_url.hostname or "127.0.0.1"
            )
            settings["chroma_server_http_port"] = settings.get(
                "chroma_server_http_port", parsed_url.port or 8000
            )
            settings["chroma_server_ssl_enabled"] = settings.get(
                "chroma_server_ssl_enabled", parsed_url.scheme == "https"
            )
            settings["chroma_server_api_default_path"] = settings.get(
                "chroma_server_api_default_path", parsed_url.path or APIVersion.V2
            )
            settings_obj = Settings(**settings)
            logger.info(
                f"Created chromadb.HttpClient from the following settings: {settings_obj}"
            )
            self._client = chromadb.HttpClient(
                host=parsed_url.hostname,
                port=parsed_url.port,
                ssl=parsed_url.scheme == "https",
                settings=settings_obj,
            )
            self._client_type = "http"
        else:
            logger.info(
                f"Created chromadb.PersistentClient at `{db_params['db_path']}` from the following settings: {settings}"
            )
            os.makedirs(db_params["db_path"], exist_ok=True)
            self._client = chromadb.PersistentClient(path=db_params["db_path"])

            self._client_type = "persistent"
        assert self._client_type in _SUPPORTED_CLIENT_TYPE
        return self._client

    async def get_client(self) -> ClientAPI:
        if self._client is None:
            self._create_client()
        assert self._client is not None
        if self._client_type == "persistent":
            async with LockManager().get_lock(
                self._configs.db_params["db_path"]
            ) as lock:
                self._lock = lock
        return self._client

    @contextlib.asynccontextmanager
    async def maybe_lock(self):
        """
        Acquire a file (dir) lock if using persistent client.
        """
        if self._lock is not None:
            await self._lock.acquire()
        yield
        if self._lock is not None:
            await self._lock.release()

    async def _create_or_get_collection(
        self, collection_path: str, allow_create: bool = False
    ) -> Collection:
        """
        This method should be used by ChromaDB methods that are expected to **create a collection when not found**.
        For other methods, just use `client.get_collection` and let it fail if the collection doesn't exist.
        """

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
        user_hnsw = db_params.get("hnsw", {})
        for key in user_hnsw.keys():
            meta_field_name: str = key
            if not meta_field_name.startswith("hnsw:"):
                meta_field_name = f"hnsw:{meta_field_name}"
            if user_hnsw.get(key) is not None:
                collection_meta[meta_field_name] = user_hnsw[key]

        async with self.maybe_lock():
            collection_id = get_collection_id(collection_path)
            client = await self.get_client()
            if not allow_create:
                try:
                    return client.get_collection(collection_id)
                except (ValueError, NotFoundError) as e:
                    raise CollectionNotFoundError(
                        f"There's no existing collection for {collection_path} in ChromaDB with the following setup: {self._configs.db_params}"
                    ) from e
            col = client.get_or_create_collection(
                collection_id, metadata=collection_meta
            )
            for key in collection_meta.keys():
                # validate metadata
                assert collection_meta[key] == col.metadata.get(key), (
                    f"Metadata field {key} mismatch!"
                )

        return col

    async def query(self) -> list[QueryResult]:
        collection = await self._create_or_get_collection(
            str(self._configs.project_root), False
        )

        assert self._configs.query is not None
        assert len(self._configs.query), "Keywords cannot be empty"
        keywords_embeddings = self.get_embedding(self._configs.query)

        query_count = self._configs.n_result or (await self.count(ResultType.chunk))
        query_filter = None
        if len(self._configs.query_exclude):
            query_filter = cast(
                chromadb.Where, {"path": {"$nin": list(self._configs.query_exclude)}}
            )
        if QueryInclude.chunk in self._configs.include:
            if query_filter is None:
                query_filter = cast(chromadb.Where, {"start": {"$gte": 0}})
            else:
                query_filter = cast(
                    chromadb.Where,
                    {"$and": [query_filter.copy(), {"start": {"$gte": 0}}]},
                )

        async with self.maybe_lock():
            raw_result = await asyncio.to_thread(
                collection.query,
                include=[
                    "metadatas",
                    "documents",
                    "distances",
                ],
                query_embeddings=keywords_embeddings,
                where=query_filter,
                n_results=query_count,
            )
        return convert_chroma_query_results(raw_result, self._configs.query)

    async def vectorise(
        self, file_path: str, chunker: TreeSitterChunker | None = None
    ) -> VectoriseStats:
        collection_path = str(self._configs.project_root)
        collection = await self._create_or_get_collection(
            collection_path, allow_create=True
        )
        chunker = chunker or TreeSitterChunker(self._configs)

        chunks = tuple(chunker.chunk(file_path))
        embeddings = self.get_embedding(list(i.text for i in chunks))
        if len(embeddings) == 0:
            return VectoriseStats(skipped=1)

        file_hash = hash_file(file_path)

        def chunk_to_meta(chunk: Chunk) -> chromadb.Metadata:
            meta: dict[str, int | str] = {"path": file_path, "sha256": file_hash}
            if chunk.start:
                meta["start"] = chunk.start.row

            if chunk.end:
                meta["end"] = chunk.end.row
            return meta

        max_bs = (await self.get_client()).get_max_batch_size()
        for batch_start_idx in range(0, len(chunks), max_bs):
            batch_chunks = [
                chunks[i].text
                for i in range(
                    batch_start_idx, min(batch_start_idx + max_bs, len(chunks))
                )
            ]
            batch_embeddings = embeddings[batch_start_idx : batch_start_idx + max_bs]
            batch_meta = [
                chunk_to_meta(chunks[i])
                for i in range(
                    batch_start_idx, min(batch_start_idx + max_bs, len(chunks))
                )
            ]
            async with self.maybe_lock():
                await asyncio.to_thread(
                    collection.add,
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_meta,
                    ids=[get_uuid() for _ in batch_chunks],
                )
        return VectoriseStats(add=1)

    async def delete(self) -> int:
        project_root = self._configs.project_root
        collection = await self._create_or_get_collection(str(project_root), False)

        rm_paths = self._configs.rm_paths
        if isinstance(rm_paths, str):
            rm_paths = [rm_paths]
        rm_paths = [
            str(expand_path(path=i, absolute=True))
            for i in await expand_globs(
                paths=self._configs.rm_paths,
                recursive=self._configs.recursive,
                include_hidden=self._configs.include_hidden,
            )
        ]

        files_in_collection = set(
            str(expand_path(i.path, True))
            for i in (
                await self.list_collection_content(what=ResultType.document)
            ).files
        )

        rm_paths = {
            str(expand_path(i, True))
            for i in rm_paths
            if os.path.isfile(i) and (i in files_in_collection)
        }

        if rm_paths:
            async with self.maybe_lock():
                collection.delete(
                    where=cast(chromadb.Where, {"path": {"$in": list(rm_paths)}})
                )
        return len(rm_paths)

    async def drop(
        self, *, collection_id: str | None = None, collection_path: str | None = None
    ):
        collection_path = str(collection_path or self._configs.project_root)
        collection_id = collection_id or get_collection_id(collection_path)
        try:
            async with self.maybe_lock():
                await asyncio.to_thread(
                    (await self.get_client()).delete_collection, collection_id
                )
        except ValueError as e:
            raise CollectionNotFoundError(
                f"Collection at {collection_path} is not found."
            ) from e

    async def get_chunks(self, file_path) -> list[Chunk]:
        file_path = os.path.abspath(file_path)
        try:
            collection = await self._create_or_get_collection(
                str(self._configs.project_root), False
            )
        except CollectionNotFoundError:
            logger.warning(
                f"There's no existing collection at {self._configs.project_root}."
            )
            return []

        raw_results = collection.get(
            where={"path": file_path},
            include=["metadatas", "documents"],
        )
        assert raw_results["metadatas"] is not None
        assert raw_results["documents"] is not None

        result: list[Chunk] = []
        for i in range(len(raw_results["ids"])):
            meta = raw_results["metadatas"][i]
            text = raw_results["documents"][i]
            _id = raw_results["ids"][i]
            chunk = Chunk(text=text, id=_id)
            if meta.get("start") is not None:
                chunk.start = Point(row=cast(int, meta["start"]), column=0)
            if meta.get("end") is not None:
                chunk.end = Point(row=cast(int, meta["end"]), column=0)

            result.append(chunk)
        return result

    async def list_collection_content(
        self,
        *,
        what: Optional[ResultType] = None,
        collection_id: str | None = None,
        collection_path: str | None = None,
    ) -> CollectionContent:
        """
        When `what` is None, this method should populate both `CollectionContent.files` and `CollectionContent.chunks`.
        Otherwise, this method may populate only one of them to save waiting time.
        """
        if collection_id is None:
            collection_path = str(collection_path or self._configs.project_root)
            collection = await self._create_or_get_collection(collection_path, False)
        else:
            try:
                collection = (await self.get_client()).get_collection(collection_id)
            except (ValueError, NotFoundError) as e:
                raise CollectionNotFoundError(
                    f"There's no existing collection for {collection_path} in ChromaDB with the following setup: {self._configs.db_params}"
                ) from e
        content = CollectionContent()
        raw_content = await asyncio.to_thread(
            collection.get,
            include=[
                "metadatas",
                "documents",
            ],
        )
        metadatas = raw_content.get("metadatas", [])
        documents = raw_content.get("documents", [])
        ids = raw_content.get("ids", [])
        assert metadatas is not None
        assert documents is not None
        assert ids is not None
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
                    start = Point(row=cast(int, metadatas[i]["start"]), column=0)
                if metadatas[i].get("end") is not None:
                    end = Point(row=cast(int, metadatas[i]["end"]), column=0)
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

    async def list_collections(self) -> Sequence[CollectionInfo]:
        client = await self.get_client()
        result: list[CollectionInfo] = []
        for col in client.list_collections():
            project_root = str(col.metadata.get("path"))
            col_counts = await self.list_collection_content(
                collection_path=project_root
            )
            result.append(
                CollectionInfo(
                    id=col.name,
                    path=project_root,
                    embedding_function=col.metadata.get(
                        "embedding_function",
                        Config().embedding_function,  # fallback to default
                    ),
                    database_backend="Chroma",
                    file_count=len(col_counts.files),
                    chunk_count=len(col_counts.chunks),
                )
            )
        return result
