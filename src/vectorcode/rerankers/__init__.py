"""VectorCode rerankers module.

This module provides reranker implementations for VectorCode.
Rerankers are used to reorder query results to improve relevance.
"""

from .base import (
    RerankerBase,
    register_reranker,
    get_reranker_class,
    list_available_rerankers,
)
from .builtins import NaiveReranker, CrossEncoderReranker
from .llama_cpp import LlamaCppReranker

# Map of legacy names to new registration names
_LEGACY_NAMES = {
    "NaiveReranker": "naive",
    "CrossEncoderReranker": "crossencoder",
    "LlamaCppReranker": "llamacpp",
}


def create_reranker(name: str, configs=None, query_chunks=None, **kwargs):
    """Create a reranker instance by name.
    
    Handles both legacy class names (e.g., 'NaiveReranker') and 
    registration names (e.g., 'naive').
    
    Args:
        name: The name of the reranker class or registered reranker
        configs: Optional Config object
        query_chunks: Optional list of query chunks for CrossEncoderReranker
        **kwargs: Additional keyword arguments to pass to the reranker
        
    Returns:
        An instance of the requested reranker
        
    Raises:
        ValueError: If the reranker name is unknown or not registered
    """
    # Check for legacy names
    registry_name = _LEGACY_NAMES.get(name, name)
    
    try:
        # Try to get class from registry
        reranker_class = get_reranker_class(registry_name)
        
        # Special case for CrossEncoderReranker which needs query_chunks
        if registry_name == "crossencoder" and query_chunks is not None:
            return reranker_class(configs=configs, query_chunks=query_chunks, **kwargs)
        else:
            return reranker_class(configs=configs, **kwargs)
            
    except ValueError:
        # Handle case where we're using a fully qualified module path
        # This is part of the dynamic import system
        raise ValueError(f"Reranker '{name}' not found in registry. "
                         f"Available rerankers: {list_available_rerankers()}")


__all__ = [
    'RerankerBase',
    'register_reranker',
    'get_reranker_class',
    'list_available_rerankers',
    'create_reranker',
    'NaiveReranker',
    'CrossEncoderReranker',
    'LlamaCppReranker',
]
