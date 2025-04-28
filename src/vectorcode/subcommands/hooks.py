import glob
import logging
import os
import platform
import re
import stat
from pathlib import Path
from typing import Optional

from vectorcode.cli_utils import GLOBAL_CONFIG_PATH, Config, find_project_root

logger = logging.getLogger(name=__name__)
__GLOBAL_HOOKS_PATH = Path(GLOBAL_CONFIG_PATH).parent / "hooks"


# Keys: name of the hooks, ie. `pre-commit`
# Values: lines of the hooks.
__HOOK_CONTENTS: dict[str, list[str]] = {
    "pre-commit": [
        "diff_files=$(git diff --cached --name-only)",
        '[ -z "$diff_files" ] || vectorcode vectorise $diff_files',
    ],
    "post-checkout": [
        'files=$(git diff --name-only "$1" "$2")',
        '[ -z "$files" ] || vectorcode vectorise $files',
    ],
}


def __lines_are_empty(lines: list[str]) -> bool:
    pattern = re.compile(r"^\s*$")
    if len(lines) == 0:
        return True
    return all(map(lambda line: pattern.match(line) is not None, lines))


def load_hooks():
    global __HOOK_CONTENTS
    for file in glob.glob(str(__GLOBAL_HOOKS_PATH / "*")):
        hook_name = Path(file).stem
        with open(file) as fin:
            lines = fin.readlines()
            if not __lines_are_empty(lines):
                __HOOK_CONTENTS[hook_name] = lines


class HookFile:
    prefix = "# VECTORCODE_HOOK_START"
    suffix = "# VECTORCODE_HOOK_END"
    prefix_pattern = re.compile(r"^\s*#\s*VECTORCODE_HOOK_START\s*")
    suffix_pattern = re.compile(r"^\s*#\s*VECTORCODE_HOOK_END\s*")

    def __init__(self, path: str | Path, git_dir: Optional[str | Path] = None):
        self.path = path
        self.lines: list[str] = []
        if os.path.isfile(self.path):
            with open(self.path) as fin:
                self.lines.extend(fin.readlines())

    def has_vectorcode_hooks(self, force: bool = False) -> bool:
        for start, start_line in enumerate(self.lines):
            if self.prefix_pattern.match(start_line) is None:
                continue

            for end in range(start + 1, len(self.lines)):
                if self.suffix_pattern.match(self.lines[end]) is not None:
                    if force:
                        logger.debug("`force` cleaning existing VectorCode hooks...")
                        new_lines = self.lines[:start] + self.lines[end + 1 :]
                        self.lines[:] = new_lines
                        return False
                    logger.debug(
                        f"Found vectorcode hook block between line {start} and {end} in {self.path}:\n{''.join(self.lines[start + 1 : end])}"
                    )
                    return True

        return False

    def inject_hook(self, content: list[str], force: bool = False):
        if len(self.lines) == 0 or not self.has_vectorcode_hooks(force):
            self.lines.append(self.prefix + "\n")
            self.lines.extend(i if i.endswith("\n") else i + "\n" for i in content)
            self.lines.append(self.suffix + "\n")
        with open(self.path, "w") as fin:
            fin.writelines(self.lines)
        if platform.system() != "Windows":
            # for unix systems, set the executable bit.
            curr_mode = os.stat(self.path).st_mode
            os.chmod(self.path, mode=curr_mode | stat.S_IXUSR)


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
