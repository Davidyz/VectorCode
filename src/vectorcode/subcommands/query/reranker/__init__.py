import logging
import sys

from vectorcode.cli_utils import Config

from .base import RerankerBase
from .cross_encoder import CrossEncoderReranker
from .naive import NaiveReranker

__all__ = ["RerankerBase", "NaiveReranker", "CrossEncoderReranker"]

logger = logging.getLogger(name=__name__)


def get_reranker(configs: Config) -> RerankerBase:
    if configs.reranker and hasattr(sys.modules[__name__], configs.reranker):
        # dynamic dispatch
        return getattr(sys.modules[__name__], configs.reranker)(configs)

    # TODO: replace the following with an Exception before the release of 0.6.0.
    logger.warning(
        f""""reranker" option should be set to one of the following: {list(i for i in __all__ if i != "RerankerBase")}.
To choose a CrossEncoderReranker model, you can set the "model_name_or_path" key in the "reranker_params" option to the name/path of the model.
To use NaiveReranker, set the "reranker" option to "NaiveReranker".
The old configuration syntax will be DEPRECATED in v0.6.0
                """
    )
    if not configs.reranker:
        return NaiveReranker(configs)
    else:
        configs.reranker_params.update({"model_name_or_path": configs.reranker})
        configs.reranker = "CrossEncoderReranker"
        return CrossEncoderReranker(
            configs,
        )
