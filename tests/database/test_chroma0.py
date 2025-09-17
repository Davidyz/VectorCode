import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from chromadb.api.types import QueryResult
from chromadb.errors import InvalidCollectionException
from tree_sitter import Point

from vectorcode.cli_utils import Config, QueryInclude
from vectorcode.database import types
from vectorcode.database.chroma0 import (
    ChromaDB0Connector,
    _convert_chroma_query_results,
)
from vectorcode.database.errors import CollectionNotFoundError


@pytest.fixture
def mock_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Config(
            project_root=tmpdir,
            db_params={
                "db_url": "http://localhost:1234",
                "db_path": os.path.join(tmpdir, "db"),
                "db_log_path": os.path.join(tmpdir, "log"),
            },
        )


@pytest.mark.asyncio
async def test_initialization(mock_config):
    """Test that the ChromaDB0Connector is initialized correctly."""
    with patch(
        "vectorcode.database.chroma0._Chroma0ClientManager.get_client",
        new_callable=AsyncMock,
    ) as mock_get_client:
        # Mock the async context manager
        mock_async_context = AsyncMock()
        mock_get_client.return_value = mock_async_context

        # Mock the client object itself
        mock_client = AsyncMock()
        mock_async_context.__aenter__.return_value = mock_client
        mock_client.get_version.return_value = "0.6.3"

        connector = ChromaDB0Connector(mock_config)
        assert connector._configs == mock_config


@pytest.mark.asyncio
async def test_query(mock_config):
    """Test the query method."""
    connector = ChromaDB0Connector(mock_config)
    connector._configs.query = ["test query"]
    connector.get_embedding = MagicMock(return_value=[[1.0, 2.0, 3.0]])

    mock_collection = AsyncMock()
    mock_collection.query.return_value = {
        "documents": [["doc1"]],
        "distances": [[0.1]],
        "metadatas": [[{"path": os.path.join(mock_config.project_root, "file1")}]],
        "ids": [["id1"]],
    }
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)

    with patch(
        "vectorcode.database.chroma0._convert_chroma_query_results"
    ) as mock_convert:
        mock_convert.return_value = ["converted_results"]
        results = await connector.query()
        assert results == ["converted_results"]
        mock_collection.query.assert_called_once()
        mock_convert.assert_called_once()


@pytest.mark.asyncio
async def test_vectorise(mock_config):
    """Test the vectorise method."""
    connector = ChromaDB0Connector(mock_config)
    mock_collection = AsyncMock()
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)

    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [MagicMock(text="chunk1")]
    connector.get_embedding = MagicMock(return_value=[[1.0, 2.0, 3.0]])

    with (
        patch("vectorcode.database.chroma0.hash_file", return_value="hash1"),
        patch("vectorcode.database.chroma0.get_uuid", return_value="uuid1"),
        patch(
            "vectorcode.database.chroma0._Chroma0ClientManager.get_client"
        ) as mock_get_client,
    ):
        mock_client = AsyncMock()
        mock_client.get_max_batch_size.return_value = 100
        mock_get_client.return_value.__aenter__.return_value = mock_client

        stats = await connector.vectorise(
            os.path.join(mock_config.project_root, "file1"), chunker=mock_chunker
        )

        assert stats.add == 1
        mock_collection.add.assert_called_once()


@pytest.mark.asyncio
async def test_list_collections(mock_config):
    """Test the list_collections method."""
    connector = ChromaDB0Connector(mock_config)
    with patch(
        "vectorcode.database.chroma0._Chroma0ClientManager.get_client"
    ) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.list_collections.return_value = ["collection1"]
        mock_collection = AsyncMock()
        mock_collection.metadata = {"path": mock_config.project_root}
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value.__aenter__.return_value = mock_client

        connector.list_collection_content = AsyncMock(
            return_value=MagicMock(files=[], chunks=[])
        )

        collections = await connector.list_collections()
        assert len(collections) == 1
        assert collections[0].id == "collection1"


@pytest.mark.asyncio
async def test_list_collection_content(mock_config):
    """Test the list_collection_content method."""
    connector = ChromaDB0Connector(mock_config)
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {
        "metadatas": [
            {
                "path": os.path.join(mock_config.project_root, "file1"),
                "sha256": "hash1",
            }
        ],
        "documents": ["doc1"],
        "ids": ["id1"],
    }
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)

    content = await connector.list_collection_content()
    assert len(content.files) == 1
    assert len(content.chunks) == 1


@pytest.mark.asyncio
async def test_delete(mock_config):
    """Test the delete method."""
    file_to_delete = os.path.join(mock_config.project_root, "file1")
    connector = ChromaDB0Connector(mock_config)
    mock_collection = AsyncMock()
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)
    connector.list_collection_content = AsyncMock(
        return_value=MagicMock(files=[MagicMock(path=file_to_delete)])
    )
    mock_config.rm_paths = [file_to_delete]

    def mock_expand_path(path, absolute):
        return path

    with (
        patch(
            "vectorcode.database.chroma0.expand_globs", return_value=[file_to_delete]
        ),
        patch("vectorcode.database.chroma0.expand_path", side_effect=mock_expand_path),
        patch("os.path.isfile", return_value=True),
    ):
        deleted_count = await connector.delete()
        assert deleted_count == 1
        mock_collection.delete.assert_called_once()


@pytest.mark.asyncio
async def test_drop(mock_config):
    """Test the drop method."""
    connector = ChromaDB0Connector(mock_config)
    with (
        patch(
            "vectorcode.database.chroma0._Chroma0ClientManager.get_client"
        ) as mock_get_client,
        patch(
            "vectorcode.database.chroma0.get_collection_id",
            return_value="collection_id",
        ),
    ):
        mock_client = AsyncMock()
        mock_get_client.return_value.__aenter__.return_value = mock_client
        await connector.drop()
        mock_client.delete_collection.assert_called_once_with("collection_id")


@pytest.mark.asyncio
async def test_get_chunks(mock_config):
    """Test the get_chunks method."""
    connector = ChromaDB0Connector(mock_config)
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {
        "metadatas": [{"start": 1, "end": 2}],
        "documents": ["doc1"],
        "ids": ["id1"],
    }
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)

    chunks = await connector.get_chunks(os.path.join(mock_config.project_root, "file1"))
    assert len(chunks) == 1
    assert chunks[0].text == "doc1"


def test_convert_chroma_query_results(mock_config):
    file1_path = os.path.join(mock_config.project_root, "file1")
    file2_path = os.path.join(mock_config.project_root, "file2")
    chroma_result: QueryResult = {
        "documents": [["doc1", "doc2"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [
            [{"path": file1_path, "start": 1, "end": 2}, {"path": file2_path}]
        ],
        "ids": [["id1", "id2"]],
        "embeddings": None,
        "uris": None,
        "data": None,
    }
    queries = ["query1"]
    results = _convert_chroma_query_results(chroma_result, queries)
    assert len(results) == 2
    assert results[0].chunk.text == "doc1"
    assert results[0].path == file1_path
    assert results[0].scores == (-0.1,)
    assert results[0].chunk.start == Point(1, 0)
    assert results[0].chunk.end == Point(2, 0)
    assert results[1].chunk.text == "doc2"
    assert results[1].path == file2_path
    assert results[1].scores == (-0.2,)


@pytest.mark.asyncio
async def test_get_chunks_collection_not_found(mock_config):
    """Test get_chunks when collection is not found."""
    connector = ChromaDB0Connector(mock_config)
    connector._create_or_get_collection = AsyncMock(side_effect=CollectionNotFoundError)
    with patch("vectorcode.database.chroma0._logger") as mock_logger:
        result = await connector.get_chunks(
            os.path.join(mock_config.project_root, "file1")
        )
        assert result == []
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_vectorise_no_embeddings(mock_config):
    """Test vectorise when there are no embeddings."""
    connector = ChromaDB0Connector(mock_config)
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [MagicMock(text="chunk1")]
    connector.get_embedding = MagicMock(return_value=[])
    with patch(
        "vectorcode.database.chroma0.ChromaDB0Connector._create_or_get_collection",
        new_callable=AsyncMock,
    ) as mock_create_collection:
        stats = await connector.vectorise(
            os.path.join(mock_config.project_root, "file1"), chunker=mock_chunker
        )
        assert stats.skipped == 1
        mock_create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_query_with_exclude(mock_config):
    """Test query with exclude paths."""
    file1_path = os.path.join(mock_config.project_root, "file1")
    file2_path = os.path.join(mock_config.project_root, "file2")
    connector = ChromaDB0Connector(mock_config)
    connector._configs.query = ["test query"]
    connector._configs.query_exclude = [file2_path]
    connector.get_embedding = MagicMock(return_value=[[1.0, 2.0, 3.0]])

    mock_collection = AsyncMock()
    mock_collection.query.return_value = {
        "documents": [["doc1"]],
        "distances": [[0.1]],
        "metadatas": [[{"path": file1_path}]],
        "ids": [["id1"]],
    }
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)

    with patch(
        "vectorcode.database.chroma0._convert_chroma_query_results"
    ) as mock_convert:
        mock_convert.return_value = ["converted_results"]
        await connector.query()
        mock_collection.query.assert_called_once()
        _, kwargs = mock_collection.query.call_args
        assert "where" in kwargs
        assert kwargs["where"] == {"path": {"$nin": [file2_path]}}


@pytest.mark.asyncio
async def test_query_with_include_chunk(mock_config):
    """Test query with include chunk."""
    connector = ChromaDB0Connector(mock_config)
    connector._configs.query = ["test query"]
    connector._configs.include = [QueryInclude.chunk]
    connector.get_embedding = MagicMock(return_value=[[1.0, 2.0, 3.0]])

    mock_collection = AsyncMock()
    mock_collection.query.return_value = {
        "documents": [["doc1"]],
        "distances": [[0.1]],
        "metadatas": [[{"path": os.path.join(mock_config.project_root, "file1")}]],
        "ids": [["id1"]],
    }
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)

    with patch(
        "vectorcode.database.chroma0._convert_chroma_query_results"
    ) as mock_convert:
        mock_convert.return_value = ["converted_results"]
        await connector.query()
        mock_collection.query.assert_called_once()
        _, kwargs = mock_collection.query.call_args
        assert "where" in kwargs
        assert kwargs["where"] == {"start": {"$gte": 0}}


@pytest.mark.asyncio
async def test_create_or_get_collection_not_found(mock_config):
    """Test _create_or_get_collection when collection is not found and allow_create is False."""
    connector = ChromaDB0Connector(mock_config)
    with patch(
        "vectorcode.database.chroma0._Chroma0ClientManager.get_client"
    ) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.get_collection.side_effect = InvalidCollectionException
        mock_get_client.return_value.__aenter__.return_value = mock_client

        with pytest.raises(CollectionNotFoundError):
            await connector._create_or_get_collection(
                "collection_path", allow_create=False
            )


@pytest.mark.asyncio
async def test_delete_no_paths(mock_config):
    """Test delete with no paths to remove."""
    file_to_keep = os.path.join(mock_config.project_root, "file1")
    non_existent_file = os.path.join(mock_config.project_root, "non_existent_file")
    connector = ChromaDB0Connector(mock_config)
    mock_collection = AsyncMock()
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)
    connector.list_collection_content = AsyncMock(
        return_value=MagicMock(files=[MagicMock(path=file_to_keep)])
    )
    mock_config.rm_paths = [non_existent_file]

    def mock_expand_path(path, absolute):
        return path

    with (
        patch(
            "vectorcode.database.chroma0.expand_globs", return_value=[non_existent_file]
        ),
        patch("vectorcode.database.chroma0.expand_path", side_effect=mock_expand_path),
        patch("os.path.isfile", return_value=True),
    ):
        deleted_count = await connector.delete()
        assert deleted_count == 0
        mock_collection.delete.assert_not_called()


@pytest.mark.asyncio
async def test_list_collection_content_with_what(mock_config):
    """Test the list_collection_content method with the 'what' parameter."""
    connector = ChromaDB0Connector(mock_config)
    mock_collection = AsyncMock()
    mock_collection.get.return_value = {
        "metadatas": [
            {
                "path": os.path.join(mock_config.project_root, "file1"),
                "sha256": "hash1",
            }
        ],
        "documents": ["doc1"],
        "ids": ["id1"],
    }
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)

    # Test with what=ResultType.document
    content = await connector.list_collection_content(what=types.ResultType.document)
    assert len(content.files) == 1
    assert len(content.chunks) == 0

    # Test with what=ResultType.chunk
    content = await connector.list_collection_content(what=types.ResultType.chunk)
    assert len(content.files) == 0
    assert len(content.chunks) == 1


@pytest.mark.asyncio
async def test_delete_with_string_rm_paths(mock_config):
    """Test delete with rm_paths as a string."""
    file_to_delete = os.path.join(mock_config.project_root, "file1")
    connector = ChromaDB0Connector(mock_config)
    mock_collection = AsyncMock()
    connector._create_or_get_collection = AsyncMock(return_value=mock_collection)
    connector.list_collection_content = AsyncMock(
        return_value=MagicMock(files=[MagicMock(path=file_to_delete)])
    )
    mock_config.rm_paths = file_to_delete

    def mock_expand_path(path, absolute):
        return path

    with (
        patch(
            "vectorcode.database.chroma0.expand_globs", return_value=[file_to_delete]
        ),
        patch("vectorcode.database.chroma0.expand_path", side_effect=mock_expand_path),
        patch("os.path.isfile", return_value=True),
    ):
        deleted_count = await connector.delete()
        assert deleted_count == 1
        mock_collection.delete.assert_called_once()


@pytest.mark.asyncio
async def test_drop_with_collection_path(mock_config):
    """Test drop with collection_path."""
    connector = ChromaDB0Connector(mock_config)
    with (
        patch(
            "vectorcode.database.chroma0._Chroma0ClientManager.get_client"
        ) as mock_get_client,
        patch(
            "vectorcode.database.chroma0.get_collection_id",
            return_value="collection_id",
        ) as mock_get_collection_id,
    ):
        mock_client = AsyncMock()
        mock_get_client.return_value.__aenter__.return_value = mock_client
        await connector.drop(collection_path=mock_config.project_root)
        mock_get_collection_id.assert_called_once_with(mock_config.project_root)
        mock_client.delete_collection.assert_called_once_with("collection_id")


@pytest.mark.asyncio
async def test_get_chunks_generic_exception(mock_config):
    """Test get_chunks with a generic exception."""
    connector = ChromaDB0Connector(mock_config)
    connector._create_or_get_collection = AsyncMock(side_effect=Exception("test error"))
    with pytest.raises(Exception) as excinfo:
        await connector.get_chunks(os.path.join(mock_config.project_root, "file1"))
    assert "test error" in str(excinfo.value)
