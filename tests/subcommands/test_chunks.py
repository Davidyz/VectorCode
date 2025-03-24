from unittest.mock import MagicMock, patch

import pytest

from vectorcode.chunking import TreeSitterChunker
from vectorcode.cli_utils import Config
from vectorcode.subcommands import chunks


@pytest.mark.asyncio
async def test_chunks():
    # Mock the Config object
    mock_config = MagicMock(spec=Config)
    mock_config.chunk_size = 2000
    mock_config.overlap_ratio = 0.2
    mock_config.files = ["file1.py", "file2.py"]

    # Mock the TreeSitterChunker
    mock_chunker = MagicMock(spec=TreeSitterChunker)
    mock_chunker.chunk.side_effect = [
        ["chunk1_file1", "chunk2_file1"],
        ["chunk1_file2", "chunk2_file2"],
    ]
    with patch(
        "vectorcode.subcommands.chunks.TreeSitterChunker", return_value=mock_chunker
    ):
        # Call the chunks function
        result = await chunks(mock_config)

        # Assertions
        assert result == 0
        mock_chunker.chunk.assert_called()
        assert mock_chunker.chunk.call_count == 2
