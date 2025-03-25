import json
import os
import sys

from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException, InvalidDimensionException

from vectorcode.chunking import StringChunker
from vectorcode.cli_utils import Config, expand_globs, expand_path
from vectorcode.common import (
    get_client,
    get_collection,
    verify_ef,
)


async def get_query_result_files(
    collection: AsyncCollection, configs: Config
) -> list[str]:
    query_chunks = []
    if configs.query:
        chunker = StringChunker(configs)
        for q in configs.query:
            query_chunks.extend(chunker.chunk(q))

    configs.query_exclude = [
        expand_path(i, True)
        for i in await expand_globs(configs.query_exclude)
        if os.path.isfile(i)
    ]
    if (await collection.count()) == 0:
        print("Empty collection!", file=sys.stderr)
        return []
    try:
        num_query = await collection.count()
        if configs.query_multiplier > 0:
            num_query = min(
                int(configs.n_result * configs.query_multiplier),
                await collection.count(),
            )
        if len(configs.query_exclude):
            filtered_files = {"path": {"$nin": configs.query_exclude}}
        else:
            filtered_files = None
        results = await collection.query(
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
        return []

    if not configs.reranker:
        from .reranker import NaiveReranker

        aggregated_results = NaiveReranker(configs).rerank(results)
    else:
        from .reranker import CrossEncoderReranker

        aggregated_results = CrossEncoderReranker(
            configs, query_chunks, configs.reranker, **configs.reranker_params
        ).rerank(results)
    return aggregated_results


async def query(configs: Config) -> int:
    client = await get_client(configs)
    try:
        collection = await get_collection(client, configs, False)
        if not verify_ef(collection, configs):
            return 1
    except (ValueError, InvalidCollectionException):
        print(
            f"There's no existing collection for {configs.project_root}",
            file=sys.stderr,
        )
        return 1
    except InvalidDimensionException:
        print(
            "The collection was embedded with a different embedding model.",
            file=sys.stderr,
        )
        return 1
    except IndexError:
        print(
            "Failed to get the collection. Please check your config.", file=sys.stderr
        )
        return 1

    if not configs.pipe:
        print("Starting querying...")

    structured_result = []

    for path in await get_query_result_files(collection, configs):
        if os.path.isfile(path):
            with open(path) as fin:
                document = fin.read()
            if configs.use_absolute_path:
                output_path = os.path.abspath(path)
            else:
                output_path = os.path.relpath(path, configs.project_root)

            full_result = {"path": output_path, "document": document}
            structured_result.append(
                {str(key): full_result[str(key)] for key in configs.include}
            )
        else:
            print(
                f"{path} is no longer a valid file! Please re-run vectorcode vectorise to refresh the database.",
                file=sys.stderr,
            )

    if configs.pipe:
        print(json.dumps(structured_result))
    else:
        for idx, result in enumerate(structured_result):
            for include_item in configs.include:
                print(f"{include_item.to_header()}{result.get(include_item.value)}")
            if idx != len(structured_result) - 1:
                print()
    return 0
