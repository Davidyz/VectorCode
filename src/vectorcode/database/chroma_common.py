from typing import Sequence, cast

from chromadb.api.types import QueryResult as ChromaQueryResult
from tree_sitter import Point

from vectorcode.chunking import Chunk
from vectorcode.database import types


def convert_chroma_query_results(
    chroma_result: ChromaQueryResult, queries: Sequence[str]
) -> list[types.QueryResult]:
    """Convert chromadb query result to in-house query results"""
    assert chroma_result["documents"] is not None
    assert chroma_result["distances"] is not None
    assert chroma_result["metadatas"] is not None
    assert chroma_result["ids"] is not None

    chroma_results_list: list[types.QueryResult] = []
    for q_i in range(len(queries)):
        q = queries[q_i]
        documents = chroma_result["documents"][q_i]
        distances = chroma_result["distances"][q_i]
        metadatas = chroma_result["metadatas"][q_i]
        ids = chroma_result["ids"][q_i]
        for doc, dist, meta, _id in zip(documents, distances, metadatas, ids):
            chunk = Chunk(text=doc, id=_id)
            if meta.get("start"):
                chunk.start = Point(cast(int, meta.get("start", 0)), 0)
            if meta.get("end"):
                chunk.end = Point(cast(int, meta.get("end", 0)), 0)
            if meta.get("path"):
                chunk.path = str(meta["path"])
            chroma_results_list.append(
                types.QueryResult(
                    chunk=chunk,
                    path=str(meta.get("path", "")),
                    query=(q,),
                    scores=(-dist,),
                )
            )
    return chroma_results_list
