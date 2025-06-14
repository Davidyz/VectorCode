from typing import Dict, Type

from vectorcode.cli_utils import Config, DbType
from vectorcode.db.base import VectorStore
from vectorcode.db.chroma import ChromaVectorStore
from vectorcode.db.local import LocalChromaVectorStore


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    _stores: Dict[DbType, Type[VectorStore]] = {
        DbType.chromadb: ChromaVectorStore,
        DbType.local: LocalChromaVectorStore,
    }

    @classmethod
    def create_store(cls, configs: Config) -> VectorStore:
        """Create a vector store instance based on configuration."""
        store_type = configs.db_type
        if store_type not in cls._stores:
            raise ValueError(f"Unsupported vector store type: {store_type}")

        store_class = cls._stores[store_type]
        return store_class(configs)
