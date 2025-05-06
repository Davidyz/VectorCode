import logging
import sys
from typing import Optional

from vectorcode.cli_utils import Config

from .base import RewriterBase
from .openai import OpenAIRewriter

logger = logging.getLogger(name=__name__)
__all__ = ["RewriterBase", "OpenAIRewriter"]


class RewriterError(Exception):
    pass


def get_rewriter(configs: Config) -> Optional[RewriterBase]:
    if configs.rewriter is None:
        logger.warning("Rewriter hasn't been configured. Skipping rewriting.")
        return None
    if configs.rewriter == "RewriterBase":
        raise RewriterError("RewriterBase is not a valid rewriter!")
    rewriter_cls = getattr(sys.modules[__name__], configs.rewriter)
    if issubclass(rewriter_cls, RewriterBase):
        logger.info(f"Loaded {configs.rewriter}")
        return rewriter_cls(configs)
    raise RewriterError(f"Failed to find {configs.rewriter}!")
