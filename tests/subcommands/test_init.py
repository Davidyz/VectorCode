import os
import tempfile
from unittest.mock import patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.subcommands.init import init


@pytest.mark.asyncio
async def test_init_new_project(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        configs = Config(project_root=temp_dir, force=False)
        return_code = await init(configs)
        assert return_code == 0
        assert os.path.isdir(os.path.join(temp_dir, ".vectorcode"))
        captured = capsys.readouterr()
        assert (
            f"VectorCode project root has been initialised at {temp_dir}"
            in captured.out
        )


@pytest.mark.asyncio
async def test_init_already_initialized(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize the project once
        configs = Config(project_root=temp_dir, force=False)
        await init(configs)

        # Try to initialize again without force
        return_code = await init(configs)
        assert return_code == 1
        captured = capsys.readouterr()
        assert f"{temp_dir} is already initialised for VectorCode." in captured.err


@pytest.mark.asyncio
async def test_init_already_initialized_with_force(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize the project once
        configs = Config(project_root=temp_dir, force=False)
        await init(configs)

        # Initialize again with force
        configs = Config(project_root=temp_dir, force=True)
        return_code = await init(configs)
        assert return_code == 0
        captured = capsys.readouterr()
        assert (
            f"VectorCode project root has been initialised at {temp_dir}"
            in captured.out
        )


@pytest.mark.asyncio
async def test_init_copies_global_config(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.path.join(temp_dir, ".vectorcode")

        # Create mock global config files
        config_items = {
            "config.json": '{"test": "value"}',
            "vectorcode.include": "*.py",
            "vectorcode.exclude": "*/tests/*",
        }

        # Patch path expansion and file operations
        with (
            patch("os.path.expanduser", return_value=temp_dir),
            patch("os.path.isfile", return_value=True),
            patch("shutil.copyfile") as copyfile_mock,
        ):
            # Create mock global config dir
            global_config_dir = os.path.join(temp_dir, ".config", "vectorcode")
            os.makedirs(global_config_dir)

            # Write mock global files
            for filename, content in config_items.items():
                with open(os.path.join(global_config_dir, filename), "w") as f:
                    f.write(content)

            # Initialize project
            configs = Config(project_root=temp_dir, force=False)
            return_code = await init(configs)

            # Assert files were copied
            assert return_code == 0
            assert copyfile_mock.call_count == len(config_items)

            # Check output messages
            captured = capsys.readouterr()
            assert (
                f"VectorCode project root has been initialised at {temp_dir}"
                in captured.out
            )
