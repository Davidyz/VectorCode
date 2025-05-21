import os
import socket
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, patch

import httpx
import pytest
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.utils import embedding_functions

from vectorcode.cli_utils import Config
from vectorcode.common import (
    get_client,
    get_collection,
    get_collection_name,
    get_collections,
    get_embedding_function,
    start_server,
    try_server,
    verify_ef,
    wait_for_server,
)


def test_get_collection_name():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_file.txt")
        collection_name = get_collection_name(file_path)
        assert isinstance(collection_name, str)
        assert len(collection_name) == 63

        # Test that the collection name is consistent for the same path
        collection_name2 = get_collection_name(file_path)
        assert collection_name == collection_name2

        # Test that the collection name is different for different paths
        file_path2 = os.path.join(temp_dir, "another_file.txt")
        collection_name2 = get_collection_name(file_path2)
        assert collection_name != collection_name2

        # Test with absolute path
        abs_file_path = os.path.abspath(file_path)
        collection_name3 = get_collection_name(abs_file_path)
        assert collection_name == collection_name3


def test_get_embedding_function():
    # Test with a valid embedding function
    config = Config(
        embedding_function="SentenceTransformerEmbeddingFunction", embedding_params={}
    )
    embedding_function = get_embedding_function(config)
    assert "SentenceTransformerEmbeddingFunction" in str(type(embedding_function))

    # Test with an invalid embedding function (fallback to SentenceTransformer)
    config = Config(embedding_function="FakeEmbeddingFunction", embedding_params={})
    embedding_function = get_embedding_function(config)
    assert "SentenceTransformerEmbeddingFunction" in str(type(embedding_function))

    # Test with specific embedding parameters
    config = Config(
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={"param1": "value1"},
    )
    embedding_function = get_embedding_function(config)
    assert "SentenceTransformerEmbeddingFunction" in str(type(embedding_function))


def test_get_embedding_function_init_exception():
    # Test when the embedding function exists but raises an error during initialization
    config = Config(
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={"model_name": "non_existent_model_should_cause_error"},
    )

    # Mock SentenceTransformerEmbeddingFunction.__init__ to raise a generic exception
    with patch.object(
        embedding_functions, "SentenceTransformerEmbeddingFunction", autospec=True
    ) as mock_stef:
        # Simulate an error during the embedding function's __init__
        mock_stef.side_effect = Exception("Simulated initialization error")

        with pytest.raises(Exception) as excinfo:
            get_embedding_function(config)

        # Check if the raised exception is the one we simulated
        assert "Simulated initialization error" in str(excinfo.value)
        # Check if the additional note was added
        assert "For errors caused by missing dependency" in excinfo.value.__notes__[0]

        # Verify that the constructor was called with the correct parameters
        mock_stef.assert_called_once_with(
            model_name="non_existent_model_should_cause_error"
        )


@pytest.mark.asyncio
async def test_try_server_versions():
    # Test successful v1 response
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )
        assert await try_server("http://localhost:8300") is True
        mock_client.return_value.__aenter__.return_value.get.assert_called_once_with(
            url="http://localhost:8300/api/v1/heartbeat"
        )

    # Test fallback to v2 when v1 fails
    with patch("httpx.AsyncClient") as mock_client:
        mock_response_v1 = MagicMock()
        mock_response_v1.status_code = 404
        mock_response_v2 = MagicMock()
        mock_response_v2.status_code = 200
        mock_client.return_value.__aenter__.return_value.get.side_effect = [
            mock_response_v1,
            mock_response_v2,
        ]
        assert await try_server("http://localhost:8300") is True
        assert mock_client.return_value.__aenter__.return_value.get.call_count == 2

    # Test both versions fail
    with patch("httpx.AsyncClient") as mock_client:
        mock_response_v1 = MagicMock()
        mock_response_v1.status_code = 404
        mock_response_v2 = MagicMock()
        mock_response_v2.status_code = 500
        mock_client.return_value.__aenter__.return_value.get.side_effect = [
            mock_response_v1,
            mock_response_v2,
        ]
        assert await try_server("http://localhost:8300") is False

    # Test connection error cases
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = (
            httpx.ConnectError("Cannot connect")
        )
        assert await try_server("http://localhost:8300") is False

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = (
            httpx.ConnectTimeout("Connection timeout")
        )
        assert await try_server("http://localhost:8300") is False


@pytest.mark.asyncio
async def test_get_client():
    # Patch chromadb.AsyncHttpClient to avoid actual network calls
    with patch("chromadb.AsyncHttpClient") as MockAsyncHttpClient:
        mock_client = MagicMock(spec=AsyncClientAPI)
        MockAsyncHttpClient.return_value = mock_client

        config = Config(db_url="https://test_host:1234", db_path="test_db")
        client = await get_client(config)

        assert isinstance(client, AsyncClientAPI)
        MockAsyncHttpClient.assert_called_once()
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].chroma_server_host
            == "test_host"
        )
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].chroma_server_http_port
            == 1234
        )
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].anonymized_telemetry
            is False
        )
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].chroma_server_ssl_enabled
            is True
        )

        # Test with valid db_settings (only anonymized_telemetry)
        config = Config(
            db_url="http://test_host1:1234",
            db_path="test_db",
            db_settings={"anonymized_telemetry": True},
        )
        client = await get_client(config)

        assert isinstance(client, AsyncClientAPI)
        MockAsyncHttpClient.assert_called()
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].chroma_server_host
            == "test_host1"
        )
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].chroma_server_http_port
            == 1234
        )
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].anonymized_telemetry
            is True
        )

        # Test with multiple db_settings, including an invalid one.  The invalid one
        # should be filtered out inside get_client.
        config = Config(
            db_url="http://test_host2:1234",
            db_path="test_db",
            db_settings={"anonymized_telemetry": True, "other_setting": "value"},
        )
        client = await get_client(config)
        assert isinstance(client, AsyncClientAPI)
        MockAsyncHttpClient.assert_called()
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].chroma_server_host
            == "test_host2"
        )
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].chroma_server_http_port
            == 1234
        )
        assert (
            MockAsyncHttpClient.call_args.kwargs["settings"].anonymized_telemetry
            is True
        )


def test_verify_ef():
    # Mocking AsyncCollection and Config
    mock_collection = MagicMock()
    mock_config = MagicMock()

    # Test when collection_ef and config.embedding_function are the same
    mock_collection.metadata = {"embedding_function": "test_embedding_function"}
    mock_config.embedding_function = "test_embedding_function"
    assert verify_ef(mock_collection, mock_config) is True

    # Test when collection_ef and config.embedding_function are different
    mock_collection.metadata = {"embedding_function": "test_embedding_function"}
    mock_config.embedding_function = "another_embedding_function"
    assert verify_ef(mock_collection, mock_config) is False

    # Test when collection_ep and config.embedding_params are the same
    mock_collection.metadata = {"embedding_params": {"param1": "value1"}}
    mock_config.embedding_params = {"param1": "value1"}
    assert verify_ef(mock_collection, mock_config) is True

    # Test when collection_ep and config.embedding_params are different
    mock_collection.metadata = {"embedding_params": {"param1": "value1"}}
    mock_config.embedding_params = {"param1": "value2"}
    assert (
        verify_ef(mock_collection, mock_config) is True
    )  # It should return True according to the source code.

    # Test when collection_ef is None
    mock_collection.metadata = {}
    mock_config.embedding_function = "test_embedding_function"
    assert verify_ef(mock_collection, mock_config) is True


@patch("socket.socket")
@pytest.mark.asyncio
async def test_try_server_mocked(mock_socket):
    # Mocking httpx.AsyncClient and its get method to simulate a successful connection
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )
        assert await try_server("http://localhost:8000") is True

    # Mocking httpx.AsyncClient to raise a ConnectError
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = (
            httpx.ConnectError("Simulated connection error")
        )
        assert await try_server("http://localhost:8000") is False

    # Mocking httpx.AsyncClient to raise a ConnectTimeout
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = (
            httpx.ConnectTimeout("Simulated connection timeout")
        )
        assert await try_server("http://localhost:8000") is False


@pytest.mark.asyncio
async def test_get_collection():
    config = Config(
        db_url="http://test_host:1234",
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
    )

    # Test retrieving an existing collection
    with patch("chromadb.AsyncHttpClient") as MockAsyncHttpClient:
        mock_client = MagicMock(spec=AsyncClientAPI)
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        MockAsyncHttpClient.return_value = mock_client

        collection = await get_collection(mock_client, config)
        assert collection == mock_collection
        mock_client.get_collection.assert_called_once()
        mock_client.get_or_create_collection.assert_not_called()

    # Test creating a collection if it doesn't exist
    with patch("chromadb.AsyncHttpClient") as MockAsyncHttpClient:
        mock_client = MagicMock(spec=AsyncClientAPI)
        mock_collection = MagicMock()

        # Clear the collection cache
        from vectorcode.common import __COLLECTION_CACHE

        __COLLECTION_CACHE.clear()

        # Make get_collection raise ValueError to trigger get_or_create_collection
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_collection.metadata = {
            "hostname": socket.gethostname(),
            "username": os.environ.get(
                "USER", os.environ.get("USERNAME", "DEFAULT_USER")
            ),
            "created-by": "VectorCode",
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        MockAsyncHttpClient.return_value = mock_client

        collection = await get_collection(mock_client, config, make_if_missing=True)
        assert collection.metadata["hostname"] == socket.gethostname()
        assert collection.metadata["username"] == os.environ.get(
            "USER", os.environ.get("USERNAME", "DEFAULT_USER")
        )
        assert collection.metadata["created-by"] == "VectorCode"
        mock_client.get_or_create_collection.assert_called_once()
        mock_client.get_collection.side_effect = None

    # Test raising IndexError on hash collision.
    with patch("chromadb.AsyncHttpClient") as MockAsyncHttpClient:
        mock_client = MagicMock(spec=AsyncClientAPI)
        mock_client.get_or_create_collection.side_effect = IndexError(
            "Hash collision occurred"
        )
        MockAsyncHttpClient.return_value = mock_client
        from vectorcode.common import __COLLECTION_CACHE

        __COLLECTION_CACHE.clear()
        with pytest.raises(IndexError):
            await get_collection(mock_client, config, make_if_missing=True)


@pytest.mark.asyncio
async def test_get_collection_hnsw():
    config = Config(
        db_url="http://test_host:1234",
        db_path="test_db",
        embedding_function="SentenceTransformerEmbeddingFunction",
        embedding_params={},
        project_root="/test_project",
        hnsw={"ef_construction": 200, "m": 32},
    )

    with patch("chromadb.AsyncHttpClient") as MockAsyncHttpClient:
        mock_client = MagicMock(spec=AsyncClientAPI)
        mock_collection = MagicMock()
        mock_collection.metadata = {
            "hostname": socket.gethostname(),
            "username": os.environ.get(
                "USER", os.environ.get("USERNAME", "DEFAULT_USER")
            ),
            "created-by": "VectorCode",
            "hnsw:ef_construction": 200,
            "hnsw:m": 32,
            "embedding_function": "SentenceTransformerEmbeddingFunction",
            "path": "/test_project",
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        MockAsyncHttpClient.return_value = mock_client

        # Clear the collection cache to force creation
        from vectorcode.common import __COLLECTION_CACHE

        __COLLECTION_CACHE.clear()

        collection = await get_collection(mock_client, config, make_if_missing=True)

        assert collection.metadata["hostname"] == socket.gethostname()
        assert collection.metadata["username"] == os.environ.get(
            "USER", os.environ.get("USERNAME", "DEFAULT_USER")
        )
        assert collection.metadata["created-by"] == "VectorCode"
        assert collection.metadata["hnsw:ef_construction"] == 200
        assert collection.metadata["hnsw:m"] == 32
        mock_client.get_or_create_collection.assert_called_once()
        assert (
            mock_client.get_or_create_collection.call_args.kwargs["metadata"]
            == mock_collection.metadata
        )


@pytest.mark.asyncio
async def test_start_server():
    with tempfile.TemporaryDirectory() as temp_dir:

        def _new_isdir(path):
            if str(temp_dir) in str(path):
                return True
            return False

        # Mock subprocess.Popen
        with (
            patch("asyncio.create_subprocess_exec") as MockCreateProcess,
            patch("asyncio.sleep"),
            patch("socket.socket") as MockSocket,
            patch("vectorcode.common.wait_for_server") as MockWaitForServer,
            patch("os.path.isdir") as mock_isdir,
            patch("os.makedirs") as mock_makedirs,
        ):
            mock_isdir.side_effect = _new_isdir
            # Mock socket to return a specific port
            mock_socket = MagicMock()
            mock_socket.getsockname.return_value = ("localhost", 12345)  # Mock port
            MockSocket.return_value.__enter__.return_value = mock_socket

            # Mock the process object
            mock_process = MagicMock()
            mock_process.returncode = 0  # Simulate successful execution
            MockCreateProcess.return_value = mock_process

            # Create a config object
            config = Config(
                db_url="http://localhost:8000",
                db_path=temp_dir,
                project_root=temp_dir,
            )

            # Call start_server
            process = await start_server(config)

            # Assert that asyncio.create_subprocess_exec was called with the correct arguments
            MockCreateProcess.assert_called_once()
            args, kwargs = MockCreateProcess.call_args
            expected_args = [
                sys.executable,
                "-m",
                "chromadb.cli.cli",
                "run",
                "--host",
                "localhost",
                "--port",
                str(12345),  # Check the mocked port
                "--path",
                temp_dir,
                "--log-path",
                os.path.join(str(config.db_log_path), "chroma.log"),
            ]
            assert args[0] == sys.executable
            assert tuple(args[1:]) == tuple(expected_args[1:])
            assert kwargs["stdout"] == subprocess.DEVNULL
            assert kwargs["stderr"] == sys.stderr
            assert "ANONYMIZED_TELEMETRY" in kwargs["env"]
            assert config.db_url == "http://127.0.0.1:12345"

            MockWaitForServer.assert_called_once_with("http://127.0.0.1:12345")

            assert process == mock_process
            mock_makedirs.assert_called_once_with(config.db_log_path)


@pytest.mark.asyncio
async def test_get_collections():
    # Mocking AsyncClientAPI and AsyncCollection
    mock_client = MagicMock(spec=AsyncClientAPI)

    # Mock successful get_collection
    mock_collection1 = MagicMock(spec=AsyncCollection)
    mock_collection1.metadata = {
        "created-by": "VectorCode",
        "username": os.environ.get("USER", os.environ.get("USERNAME", "DEFAULT_USER")),
        "hostname": socket.gethostname(),
    }

    # collection with meta == None
    mock_collection2 = MagicMock(spec=AsyncCollection)
    mock_collection2.metadata = None

    # collection with wrong "created-by"
    mock_collection3 = MagicMock(spec=AsyncCollection)
    mock_collection3.metadata = {
        "created-by": "NotVectorCode",
        "username": os.environ.get("USER", os.environ.get("USERNAME", "DEFAULT_USER")),
        "hostname": socket.gethostname(),
    }

    # collection with wrong "username"
    mock_collection4 = MagicMock(spec=AsyncCollection)
    mock_collection4.metadata = {
        "created-by": "VectorCode",
        "username": "wrong_user",
        "hostname": socket.gethostname(),
    }

    # collection with wrong "hostname"
    mock_collection5 = MagicMock(spec=AsyncCollection)
    mock_collection5.metadata = {
        "created-by": "VectorCode",
        "username": os.environ.get("USER", os.environ.get("USERNAME", "DEFAULT_USER")),
        "hostname": "wrong_host",
    }

    mock_client.list_collections.return_value = [
        "collection1",
        "collection2",
        "collection3",
        "collection4",
        "collection5",
    ]
    mock_client.get_collection.side_effect = [
        mock_collection1,
        mock_collection2,
        mock_collection3,
        mock_collection4,
        mock_collection5,
    ]

    collections = [
        collection async for collection in get_collections(mock_client)
    ]  # call get_collections
    assert len(collections) == 1
    assert collections[0] == mock_collection1


def test_get_embedding_function_fallback():
    # Test with an invalid embedding function that causes AttributeError
    config = Config(embedding_function="InvalidFunction", embedding_params={})
    embedding_function = get_embedding_function(config)
    assert "SentenceTransformerEmbeddingFunction" in str(type(embedding_function))


@pytest.mark.asyncio
async def test_wait_for_server_success():
    # Mock try_server to return True immediately
    with patch("vectorcode.common.try_server") as mock_try_server:
        mock_try_server.return_value = True

        # Should complete immediately without timeout
        await wait_for_server("http://localhost:8000", timeout=1)

        # Verify try_server was called once
        mock_try_server.assert_called_once_with("http://localhost:8000")


@pytest.mark.asyncio
async def test_wait_for_server_timeout():
    # Mock try_server to always return False
    with patch("vectorcode.common.try_server") as mock_try_server:
        mock_try_server.return_value = False

        # Should raise TimeoutError after 0.1 seconds (minimum timeout)
        with pytest.raises(TimeoutError) as excinfo:
            await wait_for_server("http://localhost:8000", timeout=0.1)

        assert "Server did not start within 0.1 seconds" in str(excinfo.value)

        # Verify try_server was called multiple times (due to retries)
        assert mock_try_server.call_count > 1
