import json

from vectorcode.chunking import TreeSitterChunker
from vectorcode.cli_utils import Config


async def chunks(configs: Config) -> int:
    chunker = TreeSitterChunker(configs.chunk_size, configs.overlap_ratio)
    result = []
    for file_path in configs.files:
        result.append(list(chunker.chunk(str(file_path))))
    print(json.dumps(result))
    return 0
