from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

class RerankerBase(ABC):
    """Base class for all rerankers in VectorCode.
    
    All rerankers should inherit from this class and implement the rerank method.
    """
    
    def __init__(self, **kwargs):
        """Initialize the reranker with kwargs.
        
        Args:
            **kwargs: Arbitrary keyword arguments to configure the reranker.
        """
        self.kwargs = kwargs
    
    @abstractmethod
    def rerank(self, results: Dict[str, Any]) -> List[str]:
        """Rerank the query results.
        
        Args:
            results: The query results from ChromaDB, typically containing ids, documents, 
                    metadatas, and distances.
                    
        Returns:
            A list of document IDs sorted in the desired order.
        """
        raise NotImplementedError("Rerankers must implement rerank method")


# Registry for reranker classes
_RERANKER_REGISTRY: Dict[str, Type[RerankerBase]] = {}


def register_reranker(name: str):
    """Decorator to register a reranker class.
    
    Args:
        name: The name to register the reranker under. This name can be used
              in configuration to specify which reranker to use.
              
    Returns:
        A decorator function that registers the decorated class.
    """
    def decorator(cls):
        _RERANKER_REGISTRY[name] = cls
        return cls
    return decorator


def get_reranker_class(name: str) -> Type[RerankerBase]:
    """Get a reranker class by name.
    
    Args:
        name: The name of the reranker class to get.
        
    Returns:
        The reranker class.
        
    Raises:
        ValueError: If the reranker name is not registered.
    """
    if name not in _RERANKER_REGISTRY:
        raise ValueError(f"Unknown reranker: {name}")
    return _RERANKER_REGISTRY[name]


def list_available_rerankers() -> List[str]:
    """List all available registered reranker names.
    
    Returns:
        A list of registered reranker names.
    """
    return list(_RERANKER_REGISTRY.keys())
