import logging
import sys
from typing import Type

from vectorcode.cli_utils import Config

from .base import RerankerBase
from .cross_encoder import CrossEncoderReranker
from .naive import NaiveReranker

__all__ = ["RerankerBase", "NaiveReranker", "CrossEncoderReranker"]

logger = logging.getLogger(name=__name__)

__supported_rerankers: dict[str, Type[RerankerBase]] = {
    "CrossEncoderReranker": CrossEncoderReranker,
    "NaiveReranker": NaiveReranker,
}


class RerankerError(Exception):
    pass


class RerankerInitialisationError(RerankerError):
    pass


def add_reranker(cls):
    """
    This is a class decorator that allows you to add a custom reranker that can be
    recognised by the `get_reranker` function.

    Your reranker should inherit `RerankerBase` and be decorated by `add_reranker`:
    ```python
    @add_reranker
    class CustomReranker(RerankerBase):
        # override the methods according to your need.
    ```
    """
    if issubclass(cls, RerankerBase):
        if __supported_rerankers.get(cls.__name__):
            error_message = f"{cls.__name__} has been registered."
            raise AttributeError(error_message)
        __supported_rerankers[cls.__name__] = cls
        return cls
    else:
        error_message = f'{cls} should be a subclass of "RerankerBase"'
        raise TypeError(error_message)


def get_available_rerankers():
    return list(__supported_rerankers.values())


def get_reranker(configs: Config) -> RerankerBase:
    if configs.reranker:
        if hasattr(sys.modules[__name__], configs.reranker):
            # dynamic dispatch for built-in rerankers
            return getattr(sys.modules[__name__], configs.reranker).create(configs)

        elif issubclass(
            __supported_rerankers.get(configs.reranker, type(None)), RerankerBase
        ):
            return __supported_rerankers[configs.reranker].create(configs)

    if not configs.reranker:
        return NaiveReranker(configs)
    else:
        raise RerankerInitialisationError()
