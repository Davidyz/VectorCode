from unittest.mock import MagicMock, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.rewriter import RewriterError, get_rewriter


def test_get_rewriter():
    assert get_rewriter(Config()) is None


def test_get_rewriter_base():
    with pytest.raises(RewriterError):
        get_rewriter(Config(rewriter="RewriterBase"))


def test_get_openai_rewriter():
    with (
        patch("vectorcode.rewriter.OpenAIRewriter") as mock_openai_cls,
        patch("vectorcode.rewriter.issubclass") as mock_issubclass,
    ):
        mock_rewriter = MagicMock()
        mock_openai_cls.return_value = mock_rewriter
        mock_issubclass.return_value = True
        assert get_rewriter(Config(rewriter="OpenAIRewriter")) == mock_rewriter


def test_get_faulty_rewriter():
    with pytest.raises(RewriterError):
        get_rewriter(Config(rewriter="DummyRewriter"))
