import logging
import os
import shutil

from vectorcode.cli_utils import Config

logger = logging.getLogger(name=__name__)


async def init(configs: Config) -> int:
    project_config_dir = os.path.join(str(configs.project_root), ".vectorcode")
    if os.path.isdir(project_config_dir) and not configs.force:
        logger.warning(
            f"{configs.project_root} is already initialised for VectorCode.",
        )
        return 1

    os.makedirs(project_config_dir, exist_ok=True)
    for item in ("config.json", "vectorcode.include", "vectorcode.exclude"):
        local_file_path = os.path.join(project_config_dir, item)
        global_file_path = os.path.join(
            os.path.expanduser("~"), ".config", "vectorcode", item
        )
        if os.path.isfile(global_file_path):
            logger.debug(f"Copying global {item} to {project_config_dir}")
            shutil.copyfile(global_file_path, local_file_path)

    print(f"VectorCode project root has been initialised at {configs.project_root}")
    print(
        "Note: The collection in the database will not be created until you vectorise a file."
    )
    return 0
