import json
import logging
import os

from chromadb import GetResult
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.errors import InvalidDimensionException, NotFoundError

from vectorcode.chunking import StringChunker
from vectorcode.cli_utils import Config, QueryInclude, expand_globs, expand_path
from vectorcode.common import (
    get_client,
    get_collection,
    verify_ef,
)

logger = logging.getLogger(name=__name__)


async def get_query_result_files(
    collection: AsyncCollection, configs: Config
) -> list[str]:
    query_chunks = []
    if configs.query:
        chunker = StringChunker(configs)
        for q in configs.query:
            query_chunks.extend(str(i) for i in chunker.chunk(q))

    configs.query_exclude = [
        expand_path(i, True)
        for i in await expand_globs(configs.query_exclude)
        if os.path.isfile(i)
    ]
    if (await collection.count()) == 0:
        logger.error("Empty collection!")
        return []
    try:
        if len(configs.query_exclude):
            logger.info(f"Excluding {len(configs.query_exclude)} files from the query.")
            filter: dict[str, dict] = {"path": {"$nin": configs.query_exclude}}
        else:
            filter = {}
        num_query = configs.n_result
        if QueryInclude.chunk in configs.include:
            filter["start"] = {"$gte": 0}
        else:
            num_query = await collection.count()
            if configs.query_multiplier > 0:
                num_query = min(
                    int(configs.n_result * configs.query_multiplier),
                    await collection.count(),
                )
                logger.info(f"Querying {num_query} chunks for reranking.")
        results = await collection.query(
            query_texts=query_chunks,
            n_results=num_query,
            include=[
                "metadatas",
                "distances",
                "documents",
            ],
            where=filter or None,
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


async def build_query_results(
    collection: AsyncCollection, configs: Config
) -> list[dict[str, str | int]]:
    structured_result = []
    for identifier in await get_query_result_files(collection, configs):
        if os.path.isfile(identifier):
            if configs.use_absolute_path:
                output_path = os.path.abspath(identifier)
            else:
                output_path = os.path.relpath(identifier, configs.project_root)
            full_result = {"path": output_path}
            with open(identifier) as fin:
                document = fin.read()
                full_result["document"] = document

            structured_result.append(
                {str(key): full_result[str(key)] for key in configs.include}
            )
        elif QueryInclude.chunk in configs.include:
            chunk: GetResult = await collection.get(
                identifier, include=["metadatas", "documents"]
            )
            meta = chunk.get(
                "metadatas",
            )
            if meta is not None and len(meta) != 0:
                full_result: dict[str, str | int] = {
                    "chunk": str(chunk.get("documents", [""])[0])
                }
                if meta[0].get("start") is not None and meta[0].get("end") is not None:
                    path = str(meta[0].get("path"))
                    with open(path) as fin:
                        start: int = meta[0]["start"]
                        end: int = meta[0]["end"]
                        full_result["chunk"] = "".join(fin.readlines()[start : end + 1])
                    full_result["start_line"] = start
                    full_result["end_line"] = end
                    if QueryInclude.path in configs.include:
                        full_result["path"] = str(
                            meta[0]["path"]
                            if configs.use_absolute_path
                            else os.path.relpath(
                                meta[0]["path"], str(configs.project_root)
                            )
                        )

                    structured_result.append(full_result)
            else:  # pragma: nocover
                logger.error(
                    "This collection doesn't support chunk-mode output because it lacks the necessary metadata. Please re-vectorise it.",
                )

        else:
            logger.warning(
                f"{identifier} is no longer a valid file! Please re-run vectorcode vectorise to refresh the database.",
            )
    return structured_result


async def query(configs: Config) -> int:
    if (
        QueryInclude.chunk in configs.include
        and QueryInclude.document in configs.include
    ):
        logger.error(
            "Having both chunk and document in the output is not supported!",
        )
        return 1
    client = await get_client(configs)
    try:
        collection = await get_collection(client, configs, False)
        if not verify_ef(collection, configs):
            return 1
    except (ValueError, NotFoundError):
        logger.error(
            f"There's no existing collection for {configs.project_root}",
        )
        return 1
    except InvalidDimensionException:
        logger.error(
            "The collection was embedded with a different embedding model.",
        )
        return 1
    except IndexError:  # pragma: nocover
        logger.error("Failed to get the collection. Please check your config.")
        return 1

    if not configs.pipe:
        print("Starting querying...")

    if QueryInclude.chunk in configs.include:
        if len((await collection.get(where={"start": {"$gte": 0}}))["ids"]) == 0:
            logger.warning(
                """
This collection doesn't contain line range metadata. Falling back to `--include path document`. 
Please re-vectorise it to use `--include chunk`.""",
            )
            configs.include = [QueryInclude.path, QueryInclude.document]

    structured_result = await build_query_results(collection, configs)

    if configs.pipe:
        print(json.dumps(structured_result))
    else:
        for idx, result in enumerate(structured_result):
            for include_item in configs.include:
                print(f"{include_item.to_header()}{result.get(include_item.value)}")
            if idx != len(structured_result) - 1:
                print()
    return 0
