import io
import json
import sys

from vectorcode.cli_utils import Config, PromptCategory
from vectorcode.subcommands import prompt


def test_prompts_pipe_true():
    configs = Config(pipe=True, prompt_categories=PromptCategory)

    # Mock stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    return_code = prompt.prompts(configs)

    sys.stdout = sys.__stdout__  # Reset stdout

    expected_output = (
        json.dumps(sorted(sum(prompt.prompt_by_categories.values(), start=[]))) + "\n"
    )
    assert captured_output.getvalue() == expected_output
    assert return_code == 0


def test_prompts_pipe_false():
    configs = Config(pipe=False, prompt_categories=PromptCategory)

    # Mock stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    return_code = prompt.prompts(configs)

    sys.stdout = sys.__stdout__  # Reset stdout

    expected_output = ""
    for i in sorted(sum(prompt.prompt_by_categories.values(), start=[])):
        expected_output += f"- {i}\n"

    assert captured_output.getvalue() == expected_output
    assert return_code == 0
