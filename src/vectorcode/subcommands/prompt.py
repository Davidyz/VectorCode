import json
import logging

from vectorcode.cli_utils import Config, PromptCategory

logger = logging.getLogger(name=__name__)

prompt_by_categories: dict[str | PromptCategory, list[str]] = {
    PromptCategory.query: [
        "separate phrases into distinct keywords when appropriate",
        "If a class, type or function has been imported from another file, this tool may be able to find its source. Add the name of the imported symbol to the query",
        "When providing answers based on VectorCode results, try to give references such as paths to files and line ranges, unless you're told otherwise (but do not include the full source code context)",
        "Avoid retrieving one single file because the retrieval mechanism may not be very accurate",
        "If the query results do not contain the needed context, increase the file count so that the result will more likely contain the desired files",
        "If the returned paths are relative, they are relative to the root of the project directory",
        "Do not suggest edits to retrieved files that are outside of the current working directory, unless the user instructed otherwise",
        "When specifying the `project_root` parameter when making a query, make sure you run the `ls` tool first to retrieve a list of valid, indexed projects",
        "If a query failed to retrieve desired results, a new attempt should use different keywords that are orthogonal to the previous ones but with similar meanings",
        "Do not use exact query keywords that you have used in a previous tool call in the conversation, unless the user instructed otherwise, or with different count/project_root",
        "Include related keywords as the search query. For example, when querying for `function`, include `return value`, `parameter`, `arguments` and alike.",
    ],
    PromptCategory.ls: [
        "Use `ls` tool to obtain a list of indexed projects that are available to be queried by the `query` command."
    ],
    PromptCategory.vectorise: [
        "When vectorising the files, provide accurate and case-sensitive paths to the file"
    ],
    "general": [
        "VectorCode is the name of this tool. Do not include it in the query unless the user explicitly asks",
        "**Use at your discretion** when you feel you don't have enough information about the repository or project",
        "**Don't escape** special characters",
    ],
}
prompt_strings = []


def prompts(configs: Config) -> int:
    results = prompt_by_categories["general"].copy()
    for item in sorted(set(configs.prompt_categories or [PromptCategory.query])):
        logger.info(f"Loading {len(prompt_by_categories[item])} prompts for {item}")
        results.extend(prompt_by_categories[item])
    results.sort()
    if configs.pipe:
        print(json.dumps(results))
    else:
        for i in results:
            print(f"- {i}")
    return 0
