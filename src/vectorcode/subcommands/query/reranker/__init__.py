import logging

from vectorcode.cli_utils import Config

from .base import RerankerBase
from .cross_encoder import CrossEncoderReranker
from .naive import NaiveReranker

__all__ = ["RerankerBase", "NaiveReranker", "CrossEncoderReranker"]

logger = logging.getLogger(name=__name__)


def get_reranker(configs: Config) -> RerankerBase:
    if configs.reranker == "NaiveReranker" or not configs.reranker:
        return NaiveReranker(configs)
    elif configs.reranker == "CrossEncoderReranker":
        return CrossEncoderReranker(configs)
    else:
        logger.warning(
            f"""
"reranker" option should be set to one of the following: {list(i for i in __all__ if i != "RerankerBase")}.
To choose a custom reranker model, you can set the "model_name_or_path" key in the "reranker_params" option to the name/path of the model.
The old configuration syntax will be deprecated in v0.6.0
                """
        )
        configs.reranker_params.update({"model_name_or_path": configs.reranker})
        configs.reranker = "CrossEncoderReranker"
        return CrossEncoderReranker(
            configs,
        )
