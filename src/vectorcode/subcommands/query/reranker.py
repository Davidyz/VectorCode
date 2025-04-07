"""
Backward compatibility module for rerankers.

This module re-exports the reranker classes from the new vectorcode.rerankers module
to maintain backward compatibility with existing code.

For new code, please use the vectorcode.rerankers module directly.
"""

import sys
import warnings

# Emit a deprecation warning
warnings.warn(
    "The vectorcode.subcommands.query.reranker module is deprecated. "
    "Please use vectorcode.rerankers instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from the new module
from vectorcode.rerankers import (
    RerankerBase,
    NaiveReranker,
    CrossEncoderReranker,
)

# Make sure we export all the classes
__all__ = [
    'RerankerBase',
    'NaiveReranker',
    'CrossEncoderReranker',
]
