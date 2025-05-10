import logging
import os

from vectorcode.cli_utils import Config, find_project_root
from vectorcode.subcommands.init import __HOOK_CONTENTS, HookFile, load_hooks

logger = logging.getLogger(name=__name__)


async def hooks(configs: Config) -> int:
    project_root = configs.project_root or "."
    git_root = find_project_root(project_root, ".git")
    if git_root is None:
        logger.error(f"{project_root} is not inside a git repo directory!")
        return 1
    load_hooks()
    for hook in __HOOK_CONTENTS.keys():
        hook_file_path = os.path.join(git_root, ".git", "hooks", hook)
        logger.info(f"Writing {hook} hook into {hook_file_path}.")
        print(f"Processing {hook} hook...")
        hook_obj = HookFile(hook_file_path, git_dir=git_root)
        hook_obj.inject_hook(__HOOK_CONTENTS[hook], configs.force)
    return 0
