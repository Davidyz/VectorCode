from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from pydantic_core import InitErrorDetails

from vectorcode.cli_utils import Config
from vectorcode.rewriter.openai import OpenAIRewriter, _NewQuery


class MockChatCompletionMessageParsed:
    def __init__(self, keywords):
        self.keywords = keywords


class MockChatCompletionMessage:
    def __init__(self, content, parsed=None, refusal=None):
        self.content = content
        self.parsed = parsed
        self.refusal = refusal


class MockChatCompletionChoice:
    def __init__(self, message):
        self.message = message


class MockChatCompletion:
    def __init__(self, choices):
        self.choices = choices
        # Add other minimal required attributes if needed by the method
        self.id = "mock-id"
        self.created = 1234567890
        self.model = "mock-model"
        self.object = "chat.completion"
        self.system_fingerprint = None


@pytest.fixture(scope="function")
def client():
    # Patch the actual openai.Client during the test
    with patch("vectorcode.rewriter.openai.openai.Client") as mock_client:
        # Configure the beta.chat.completions.parse part of the mock
        mock_client.return_value.beta.chat.completions.parse = MagicMock()
        yield mock_client.return_value  # Yield the instance of the mock client


def test_openai_config_setup():
    config = Config(
        rewriter="OpenAIRewriter",
        rewriter_params={
            "client_kwargs": {"test_arg": "test_val"},
            "system_prompt": "hello world",
        },
    )
    with patch("vectorcode.rewriter.openai.openai.Client") as mock_client:
        rewriter = OpenAIRewriter(config)
        assert rewriter.system_prompt == "hello world"
        # Check if the client was initialized with the correct kwargs
        # The fixture yields the *instance*, so we check the call on the patched class
        mock_client.assert_called_once_with(**config.rewriter_params["client_kwargs"])


@pytest.mark.asyncio
async def test_openai_rewrite_fallback_empty_response(client: MagicMock):
    """Tests fallback to original query when LLM returns None or empty choices."""
    # Case 1: LLM returns None
    client.beta.chat.completions.parse.return_value = None
    config = Config(query=["test_query_none"])
    rewriter = OpenAIRewriter(config)
    assert await rewriter.rewrite(["test_query_none"]) == ["test_query_none"], (
        "Should fallback when LLM returns None"
    )

    # Case 2: LLM returns a completion with no choices
    mock_comp_no_choices = MockChatCompletion(choices=[])
    client.beta.chat.completions.parse.return_value = mock_comp_no_choices
    config = Config(query=["test_query_no_choices"])
    rewriter = OpenAIRewriter(config)
    assert await rewriter.rewrite(["test_query_no_choices"]) == [
        "test_query_no_choices"
    ], "Should fallback when LLM returns empty choices"


@pytest.mark.asyncio
async def test_openai_rewrite_success(client: MagicMock):
    """Tests successful rewriting and parsing of the LLM response."""
    original_query = ["Pytoch", "train model"]
    rewritten_keywords = ["keyword1", "keyword2"]
    mock_parsed = MockChatCompletionMessageParsed(keywords=rewritten_keywords)
    mock_message = MockChatCompletionMessage(
        content="mock json", parsed=mock_parsed, refusal=None
    )
    mock_choice = MockChatCompletionChoice(message=mock_message)
    mock_comp_success = MockChatCompletion(choices=[mock_choice])

    client.beta.chat.completions.parse.return_value = mock_comp_success

    # Use a config that might include completion_kwargs
    config = Config(
        rewriter="OpenAIRewriter",
        rewriter_params={"completion_kwargs": {"model": "gpt-4-test"}},
    )
    rewriter = OpenAIRewriter(config)

    result = await rewriter.rewrite(original_query)

    assert result == rewritten_keywords, "Should return the parsed keywords on success"

    # Verify the client method was called with the correct arguments
    client.beta.chat.completions.parse.assert_called_once_with(
        messages=[
            {"role": "system", "content": rewriter.system_prompt},
            {"role": "user", "content": " ".join(original_query)},
        ],
        response_format=_NewQuery,
        model="gpt-4-test",  # Check kwargs passed from config
    )


@pytest.mark.asyncio
async def test_openai_rewrite_no_parsed_content(client: MagicMock):
    """Tests fallback when the LLM response has a choice but no parsed content (e.g., refusal)."""
    original_query = ["test query refusal"]
    mock_message = MockChatCompletionMessage(
        content="I cannot process that.", parsed=None, refusal="Refusal reason"
    )
    mock_choice = MockChatCompletionChoice(message=mock_message)
    mock_comp_refusal = MockChatCompletion(choices=[mock_choice])

    client.beta.chat.completions.parse.return_value = mock_comp_refusal

    config = Config(query=original_query)
    rewriter = OpenAIRewriter(config)

    result = await rewriter.rewrite(original_query)

    assert result == original_query, (
        "Should fallback when the LLM response has no parsed content"
    )
    client.beta.chat.completions.parse.assert_called_once()  # Check it was called


@pytest.mark.asyncio
async def test_openai_rewrite_validation_error(client: MagicMock):
    """Tests fallback when parsing the LLM response raises a ValidationError."""
    original_query = ["test query validation error"]

    def raise_validation_error(*args, **kwargs):
        errors = [
            {
                "type": "value_error",
                "input": "invalid input",
                "ctx": {"error": "value_error"},
            }
        ]
        raise ValidationError.from_exception_data(
            "foobar", cast(list[InitErrorDetails], errors)
        )

    client.beta.chat.completions.parse.side_effect = raise_validation_error

    config = Config(query=original_query)
    rewriter = OpenAIRewriter(config)

    result = await rewriter.rewrite(original_query)

    assert result == original_query, (
        "Should fallback when a ValidationError occurs during parsing"
    )
    client.beta.chat.completions.parse.assert_called_once()
