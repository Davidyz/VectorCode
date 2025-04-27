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
            logger.error(error_message)
            raise AttributeError(error_message)
        __supported_rerankers[cls.__name__] = cls
        return cls
    else:
        error_message = f'{cls} should be a subclass of "RerankerBase"'
        logger.error(error_message)
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

    # TODO: replace the following with an Exception before the release of 0.6.0.
    logger.warning(
        f""""reranker" option should be set to one of the following: {list(i.__name__ for i in get_available_rerankers())}.
To choose a CrossEncoderReranker model, you can set the "model_name_or_path" key in the "reranker_params" option to the name/path of the model.
To use NaiveReranker, set the "reranker" option to "NaiveReranker".
The old configuration syntax will be DEPRECATED in v0.6.0
                """
    )
    if not configs.reranker:
        return NaiveReranker(configs)
    else:
        logger.error(f"{configs.reranker} is not a valid reranker type!")
        raise RerankerInitialisationError()
