import json
import os

from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException, InvalidDimensionException

from vectorcode.chunking import StringChunker
from vectorcode.cli_utils import Config, expand_globs, expand_path
from vectorcode.common import (
    get_client,
    get_collection_name,
    get_embedding_function,
    verify_ef,
)

from .reranker import ArithmeticMeanReranker


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
            include=[IncludeEnum.metadatas, IncludeEnum.distances],
            where=filtered_files,
        )
    except IndexError:
        # no results found
        return 0

    structured_result = []
    aggregated_results = ArithmeticMeanReranker(configs).rerank(results)
    for path in aggregated_results:
        with open(path) as fin:
            document = fin.read()
        structured_result.append({"path": path, "document": document})

    if configs.pipe:
        print(json.dumps(structured_result))
    else:
        for idx, result in enumerate(structured_result):
            print(f"Path: {result['path']}")
            print(f"Content: \n{result['document']}")
            if idx != len(structured_result) - 1:
                print()
    return 0
