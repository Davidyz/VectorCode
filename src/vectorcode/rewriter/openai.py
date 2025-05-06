import logging
from typing import override

import openai
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from vectorcode.cli_utils import Config
from vectorcode.rewriter.base import RewriterBase

logger = logging.getLogger(name=__name__)


class _NewQuery(BaseModel):
    keywords: list[str] = Field(
        description="Orthogonal keywords for the vector search."
    )


class OpenAIRewriter(RewriterBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.client = openai.Client(
            **self.config.rewriter_params.get("client_kwargs", {})
        )
        self.system_prompt = """
Role:
        You are a code-aware rewriter that improves technical queries/docs for retrieval. Never assume a programming language unless the input explicitly includes syntax, APIs, or error messages from one.
Rules:

    For Queries:

        Fix unambiguous typos (e.g., "Pytoch" → "PyTorch").
        
        Omit langauge-specific keywords (e.g., "async def foo():" → "foo")

    For Docs/Code:

        Clarify ambiguous terms only with explicit context.

        Never modify code logic or variable names.

Anti-Goals:

    No language assumptions.

    No code changes.

    No hallucinations.
"""

    @override
    async def rewrite(self, original_query: list[str]):
        comp: ChatCompletion = self.client.beta.chat.completions.parse(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": " ".join(original_query)},
            ],
            response_format=_NewQuery,
            **self.config.rewriter_params.get("completion_kwargs", {}),
        )
        if comp is None or len(comp.choices) == 0:
            logger.info("Recieved no rewritten query. Fallingback to original_query.")
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
