import json
import logging
import os

from vectorcode.chunking import Chunk, StringChunker
from vectorcode.cli_utils import (
    Config,
    QueryInclude,
)
from vectorcode.database import get_database_connector
from vectorcode.database.base import DatabaseConnectorBase
from vectorcode.subcommands.query.reranker import (
    get_reranker,
)

logger = logging.getLogger(name=__name__)


def _prepare_formatted_result(
    reranked_results: list[str | Chunk],
) -> list[dict[str, str | int]]:
    results: list[dict[str, str | int]] = []
    for res in reranked_results:
        if isinstance(res, str):
            if os.path.isfile(res):
                # path to a file
                with open(res) as fin:
                    results.append({"path": res, "document": fin.read()})
            else:  # pragma: nocover
                logger.warning(f"Skipping non-existent file: {res}")
        else:
            assert isinstance(res, Chunk)
            if res.start is None or res.end is None:
                logger.warning(
                    "This chunk doesn't have line range metadata. Please try re-vectorising the project."
                )
            output_dict = {
                "path": res.path,
                "chunk": res.text,
                "end_line": res.end.row if res.end is not None else None,
                "chunk_id": res.id,
            }
            if res.start:
                output_dict["start_line"] = res.start.row
            if res.end:
                output_dict["end_line"] = res.end.row
            results.append(output_dict)
    return results


async def get_reranked_results(
    config: Config,
    database: DatabaseConnectorBase,
) -> list[str | Chunk]:
    """
    Return a list of paths or `Chunk`s ranked by similarity.
    """
    reranker = get_reranker(config)
    reranked_results = await reranker.rerank(results=await database.query())
    return reranked_results


def preprocess_query_keywords(configs: Config):
    assert configs.query
    query_chunks: list[str] = []
    chunker = StringChunker(configs)
    for q in configs.query:
        query_chunks.extend(str(i) for i in chunker.chunk(q))
    configs.query[:] = query_chunks
    return configs


def verify_include(configs: Config):
    if QueryInclude.path not in configs.include:
        configs.include.append(QueryInclude.path)
    assert not (
        QueryInclude.chunk in configs.include
        and QueryInclude.document in configs.include
    ), "`chunk` and `document` cannot be used at the same time for `--include`."


async def query(configs: Config) -> int:
    verify_include(configs)

    assert configs.query
    preprocess_query_keywords(configs)

    database = get_database_connector(configs)
    reranked_results = await get_reranked_results(configs, database)
    formatted_results = _prepare_formatted_result(reranked_results)
    if configs.pipe:
        print(json.dumps(formatted_results))
    else:
        for idx, result in enumerate(formatted_results):
            for include_item in configs.include:
                print(f"{include_item.to_header()}{result.get(include_item.value)}")
            if idx != len(formatted_results) - 1:
                print()
    return 0
