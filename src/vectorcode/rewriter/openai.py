import logging

import openai
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, ValidationError

from vectorcode.cli_utils import Config
from vectorcode.rewriter.base import RewriterBase

logger = logging.getLogger(name=__name__)


class _NewQuery(BaseModel):
    keywords: list[str] = Field(
        description="Orthogonal keywords for the vector search."
    )


class OpenAIRewriter(RewriterBase):
    """
    OpenAIRewriter class is an adapter for openai-compatible API services that provides
    structured output support. The `rewriter_params` dictionary accepts 3 keys:
        - `client_kwargs`: dictionary, containing arguments that are passed to `openai.Client`.
          See https://github.com/openai/openai-python/blob/67997a4ec1ebcdf8e740afb0d0b2e37897657bde/src/openai/_client.py#L80;
        - `completion_kwargs`: dictionary, containing arguments that are passed to `openai.Client.beta.chat.completions.parse`.
          See https://github.com/openai/openai-python/blob/main/helpers.md#structured-outputs-parsing-helpers.
        - `system_prompt`: string, the system prompt that contains the guidelines for rewriting the query.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.client = openai.Client(
            **self.config.rewriter_params.get("client_kwargs", {})
        )
        self.system_prompt = self.config.rewriter_params.get(
            "system_prompt",
            """
Role:
        You are a code-aware rewriter that improves technical queries/docs for retrieval. Never assume a programming language unless the input explicitly includes syntax, APIs, or error messages from one.
Rules:

    For Queries:

        Fix unambiguous typos (e.g., "Pytoch" → "PyTorch").
        
        Omit langauge-specific keywords (e.g., "async def foo():" → "foo")

        Do not include standard libraries the query.

    For Docs/Code:

        Clarify ambiguous terms only with explicit context.

        Never modify code logic or variable names.

Anti-Goals:

    No language assumptions.

    No code changes.

    No hallucinations.
""",
        )

    async def rewrite(self, original_query: list[str]):
        try:
            comp: ChatCompletion = self.client.beta.chat.completions.parse(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": " ".join(original_query)},
                ],
                response_format=_NewQuery,
                **self.config.rewriter_params.get("completion_kwargs", {}),
            )
            if comp is None or len(comp.choices) == 0:
                logger.info(
                    "Received no rewritten query. Fallingback to original_query."
                )
                return original_query
            choice = comp.choices[0].message
            if choice and choice.parsed:
                logger.debug(f"Rewritten queries to: {choice.parsed}")
                return choice.parsed.keywords
            else:
                logger.warning(
                    f"Failed to parse structured output: {choice.refusal}. Fallingback to original_query."
                )
                return original_query
        except ValidationError:
            logger.warning(
                "Failed to parse structured output. Fallingback to original_query."
            )
            return original_query
