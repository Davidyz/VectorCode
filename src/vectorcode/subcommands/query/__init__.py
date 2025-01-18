import json
import os
import sys
from collections import defaultdict
from enum import Enum
from typing import Callable, DefaultDict

import numpy
from chromadb.api.types import IncludeEnum, QueryResult
from chromadb.errors import InvalidCollectionException, InvalidDimensionException

from vectorcode.chunking import StringChunker
from vectorcode.cli_utils import Config, expand_globs, expand_path
from vectorcode.common import (
    get_client,
    get_collection_name,
    get_embedding_function,
    verify_ef,
)


class TopKStrategy(Enum):
    # TODO: make this configurable
    a_mean: Callable[..., float] = lambda x: float(numpy.mean(x))
    g_mean: Callable[..., float] = lambda x: float(numpy.exp(numpy.log(x).mean()))
    min: Callable[..., float] = lambda x: min(x)


def top_k_results(results: QueryResult, configs: Config) -> list[str]:
    assert results["metadatas"] is not None
    assert results["distances"] is not None
    documents: DefaultDict[str, list[float]] = defaultdict(list)
    for query_chunk_idx in range(len(results["ids"])):
        chunk_metas = results["metadatas"][query_chunk_idx]
        chunk_distances = results["distances"][query_chunk_idx]
        paths = [str(meta["path"]) for meta in chunk_metas]
        assert len(paths) == len(chunk_distances)
        for distance, path in zip(chunk_distances, paths):
            documents[path].append(distance)

    doc_to_dist_arr = {}
    for key in documents.keys():
        doc_to_dist_arr[key] = numpy.array(documents[key])
    doc_list = sorted(
        doc_to_dist_arr.keys(),
        key=lambda x: TopKStrategy.min(doc_to_dist_arr[x]),
    )
    return doc_list[: configs.n_result]


def query(configs: Config) -> int:
    client = get_client(configs)
    try:
        collection = client.get_collection(
            name=get_collection_name(str(configs.project_root)),
            embedding_function=get_embedding_function(configs),
        )
        if not verify_ef(collection, configs):
            return 1
    except (ValueError, InvalidCollectionException):
        print(f"There's no existing collection for {configs.project_root}")
        return 1
    except InvalidDimensionException:
        print("The collection was embedded with a different embedding model.")
        return 1

    if not configs.pipe:
        print("Starting querying...")

    query_chunks = list(
        StringChunker(configs.chunk_size, configs.overlap_ratio).chunk(
            configs.query or ""
        )
    )
    configs.query_exclude = [
        expand_path(i, True)
        for i in expand_globs(configs.query_exclude)
        if os.path.isfile(i)
    ]
    try:
        num_query = collection.count()
        if configs.query_multiplier > 0:
            num_query = configs.n_result * configs.query_multiplier
        if len(configs.query_exclude):
            filtered_files = {"path": {"$nin": configs.query_exclude}}
        else:
            filtered_files = None
        results = collection.query(
            query_texts=query_chunks,
            n_results=num_query,
            include=[
                IncludeEnum.metadatas,
                IncludeEnum.distances,
                IncludeEnum.documents,
            ],
            where=filtered_files,
        )
    except IndexError:
        # no results found
        return 0

    structured_result = []
    if configs.reranker is None:
        from .reranker import NaiveReranker

        aggregated_results = NaiveReranker(configs).rerank(results)
    else:
        from .reranker import FlagEmbeddingReranker

        aggregated_results = FlagEmbeddingReranker(
            configs, query_chunks, configs.reranker
        ).rerank(results)
    for path in aggregated_results:
        try:
            with open(path) as fin:
                document = fin.read()
            structured_result.append({"path": path, "document": document})
        except FileNotFoundError:
            print(
                f"{path} is no longer a valid file! Please re-run vectorcode vectorise to refresh the database.",
                file=sys.stderr,
            )

    if configs.pipe:
        print(json.dumps(structured_result))
    else:
        for idx, result in enumerate(structured_result):
            print(f"Path: {result['path']}")
            print(f"Content: \n{result['document']}")
            if idx != len(structured_result) - 1:
                print()
    return 0
