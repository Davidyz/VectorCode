import json

from vectorcode.chunking import TreeSitterChunker
from vectorcode.cli_utils import Config


async def chunks(configs: Config) -> int:
    chunker = TreeSitterChunker(configs)
    result = []
    for file_path in configs.files:
        result.append(list(i.text for i in chunker.chunk(str(file_path))))
    print(json.dumps((result)))
    return 0
