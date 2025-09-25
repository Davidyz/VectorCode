import hashlib
import logging
import os
import socket
import uuid
from functools import cache

import chromadb
from chromadb.utils import embedding_functions

from vectorcode.cli_utils import Config, expand_path

logger = logging.getLogger(name=__name__)


def hash_str(string: str) -> str:
    """Return the sha-256 hash of a string."""
    return hashlib.sha256(string.encode()).hexdigest()


def hash_file(path: str) -> str:
    """return the sha-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(8192)
            if chunk:
                hasher.update(chunk)
            else:
                break
    return hasher.hexdigest()


def get_uuid() -> str:
    return uuid.uuid4().hex


def get_collection_id(full_path: str) -> str:
    full_path = str(expand_path(full_path, absolute=True))
    hasher = hashlib.sha256()
    plain_collection_name = f"{os.environ.get('USER', os.environ.get('USERNAME', 'DEFAULT_USER'))}@{socket.gethostname()}:{full_path}"
    hasher.update(plain_collection_name.encode())
    collection_id = hasher.hexdigest()[:63]
    logger.debug(
        f"Hashing {plain_collection_name} as the collection name for {full_path}."
    )
    return collection_id


@cache
def get_embedding_function(
    configs: Config,
) -> chromadb.EmbeddingFunction:  # pragma: nocover
    try:
        ef = getattr(embedding_functions, configs.embedding_function)(
            **configs.embedding_params
        )
        if ef is None:  # pragma: nocover
            raise AttributeError()
        return ef
    except AttributeError:
        logger.warning(
            f"Failed to use {configs.embedding_function}. Falling back to Sentence Transformer.",
        )
        return embedding_functions.SentenceTransformerEmbeddingFunction()  # type:ignore
    except Exception as e:
        e.add_note(
            "\nFor errors caused by missing dependency, consult the documentation of pipx (or whatever package manager that you installed VectorCode with) for instructions to inject libraries into the virtual environment."
        )
        logger.error(
            f"Failed to use {configs.embedding_function} with following error.",
        )
        raise
