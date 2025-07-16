import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from vectorcode.cli_utils import Config
from vectorcode.subcommands.init import HookFile, __lines_are_empty, init, load_hooks


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
        assert return_code != 0


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
            "config.json5": '{"test": "value"}',
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


@pytest.mark.asyncio
async def test_hooks_orchestration_with_hooks():
    """Test hooks orchestration: handles git repo and loaded hooks."""

    with tempfile.TemporaryDirectory() as temp_dir:
        mock_config = Config(project_root=os.path.join(temp_dir, "project"), hooks=True)
        defined_hooks = {
            "pre-commit": ["line1"],
            "post-commit": ["lineA", "lineB"],
        }

        mock_hook_instance = MagicMock()

        with (
            patch.dict(
                "vectorcode.subcommands.init.__HOOK_CONTENTS", defined_hooks, clear=True
            ),
            patch("vectorcode.subcommands.init.HookFile") as mock_HookFile,
            patch(
                "vectorcode.subcommands.init.find_project_root",
                return_value=os.path.join(temp_dir, "project"),
            ),
            patch("vectorcode.subcommands.init.load_hooks") as mock_load_hooks,
        ):
            mock_HookFile.return_value = mock_hook_instance
            return_code = await init(mock_config)

            mock_load_hooks.assert_called_once()
            assert return_code == 0


@pytest.fixture(scope="function")
def mock_hook_path() -> Path:
    return Path("/fake/git/repo/.git/hooks/pre-commit")


@pytest.fixture(autouse=True, scope="function")
def reset_hook_contents():
    from vectorcode.subcommands.init import __GLOBAL_HOOKS_PATH, __HOOK_CONTENTS

    original_hooks_path = __GLOBAL_HOOKS_PATH
    original_contents = __HOOK_CONTENTS.copy()
    __HOOK_CONTENTS.clear()

    __GLOBAL_HOOKS_PATH = Path("/tmp/fake/global/hooks")
    yield

    __HOOK_CONTENTS = original_contents
    __GLOBAL_HOOKS_PATH = original_hooks_path


def test_lines_are_empty():
    assert __lines_are_empty([])
    assert __lines_are_empty([""])
    assert __lines_are_empty([" ", "\t"])
    assert not __lines_are_empty([" hello ", "\tworld"])


@patch("vectorcode.subcommands.init.glob")
@patch("vectorcode.subcommands.open", new_callable=mock_open)
def test_load_hooks_no_files(mock_open_func, mock_glob):
    from vectorcode.subcommands.init import __GLOBAL_HOOKS_PATH

    mock_glob.glob.return_value = []
    expected_glob_path = str(__GLOBAL_HOOKS_PATH / "*")

    load_hooks()

    mock_glob.glob.assert_called_once_with(expected_glob_path)
    mock_open_func.assert_not_called()


@patch("vectorcode.subcommands.init.glob")
@patch(
    "vectorcode.subcommands.init.open",
    new_callable=mock_open,
    read_data="Hook line 1\nLine 2",
)
def test_load_hooks_one_file(mock_open_func, mock_glob):
    """Test load_hooks with a single valid hook file."""
    from vectorcode.subcommands.init import __GLOBAL_HOOKS_PATH, __HOOK_CONTENTS

    hook_file_path = str(__GLOBAL_HOOKS_PATH / "test-hook")
    mock_glob.glob.return_value = [hook_file_path]
    expected_glob_path = str(__GLOBAL_HOOKS_PATH / "*")

    load_hooks()

    assert "test-hook" in __HOOK_CONTENTS
    assert __HOOK_CONTENTS["test-hook"] == ["Hook line 1\n", "Line 2"]
    mock_glob.glob.assert_called_once_with(expected_glob_path)
    mock_open_func.assert_called_once_with(hook_file_path)


@patch("vectorcode.subcommands.init.glob")
@patch("vectorcode.subcommands.init.open", new_callable=mock_open)
def test_load_hooks_multiple_files(mock_open_func, mock_glob):
    from vectorcode.subcommands.init import __GLOBAL_HOOKS_PATH, __HOOK_CONTENTS

    """Test load_hooks with multiple hook files."""

    hook_file_path1 = str(__GLOBAL_HOOKS_PATH / "pre-commit")
    hook_file_path2 = str(__GLOBAL_HOOKS_PATH / "post-commit.sh")
    mock_glob.glob.return_value = [hook_file_path1, hook_file_path2]
    expected_glob_path = str(__GLOBAL_HOOKS_PATH / "*")

    mock_open_func.side_effect = [
        mock_open(read_data="Pre-commit content\n").return_value,
        mock_open(read_data="Post-commit content\n").return_value,
    ]

    load_hooks()

    assert len(__HOOK_CONTENTS) == 2
    assert "pre-commit" in __HOOK_CONTENTS
    assert "post-commit" in __HOOK_CONTENTS
    assert __HOOK_CONTENTS["pre-commit"] == ["Pre-commit content\n"]
    assert __HOOK_CONTENTS["post-commit"] == ["Post-commit content\n"]
    mock_glob.glob.assert_called_once_with(expected_glob_path)
    assert mock_open_func.call_count == 2
    mock_open_func.assert_any_call(hook_file_path1)
    mock_open_func.assert_any_call(hook_file_path2)


@patch("vectorcode.subcommands.init.glob")
@patch("vectorcode.subcommands.init.open", new_callable=mock_open, read_data="")
def test_load_hooks_empty_file(mock_open_func, mock_glob):
    from vectorcode.subcommands.init import __GLOBAL_HOOKS_PATH, __HOOK_CONTENTS

    """Test load_hooks with an empty hook file."""

    hook_file_path = str(__GLOBAL_HOOKS_PATH / "empty-hook")
    mock_glob.glob.return_value = [hook_file_path]
    expected_glob_path = str(__GLOBAL_HOOKS_PATH / "*")

    load_hooks()

    assert not __HOOK_CONTENTS
    mock_glob.glob.assert_called_once_with(expected_glob_path)
    mock_open_func.assert_called_once_with(hook_file_path)


@patch("vectorcode.subcommands.init.glob")
@patch(
    "vectorcode.subcommands.init.open", new_callable=mock_open, read_data="\n   \n\t\n"
)
def test_load_hooks_whitespace_file(mock_open_func, mock_glob):
    """Test load_hooks with a hook file containing only whitespace."""
    from vectorcode.subcommands.init import __GLOBAL_HOOKS_PATH, __HOOK_CONTENTS

    hook_file_path = str(__GLOBAL_HOOKS_PATH / "whitespace-hook")
    mock_glob.glob.return_value = [hook_file_path]
    expected_glob_path = str(__GLOBAL_HOOKS_PATH / "*")

    load_hooks()

    assert not __HOOK_CONTENTS
    mock_glob.glob.assert_called_once_with(expected_glob_path)
    mock_open_func.assert_called_once_with(hook_file_path)


@patch("vectorcode.subcommands.init.os.path.isfile")
@patch(
    "vectorcode.subcommands.init.open",
    new_callable=mock_open,
    read_data="Existing line 1\nExisting line 2",
)
def test_hookfile_init_existing_file(mock_open_func, mock_isfile, mock_hook_path):
    """Test HookFile initialization when the hook file exists."""
    mock_isfile.return_value = True

    hook_file = HookFile(mock_hook_path)

    mock_isfile.assert_called_once_with(mock_hook_path)
    mock_open_func.assert_called_once_with(mock_hook_path)
    assert hook_file.path == mock_hook_path
    assert hook_file.lines == ["Existing line 1\n", "Existing line 2"]


@patch("vectorcode.subcommands.init.os.path.isfile")
@patch("vectorcode.subcommands.init.open", new_callable=mock_open)
def test_hookfile_init_non_existent_file(mock_open_func, mock_isfile, mock_hook_path):
    """Test HookFile initialization when the hook file does not exist."""
    mock_isfile.return_value = False

    hook_file = HookFile(mock_hook_path)

    mock_isfile.assert_called_once_with(mock_hook_path)
    mock_open_func.assert_not_called()
    assert hook_file.path == mock_hook_path
    assert hook_file.lines == []


@pytest.mark.parametrize(
    "lines, expected",
    [
        ([], False),
        (["Some content"], False),
        ([HookFile.prefix + "\n"], False),
        ([HookFile.suffix + "\n"], False),
        ([HookFile.prefix + "\n", HookFile.suffix + "\n"], True),
        (
            [
                "Line 1\n",
                HookFile.prefix + "\n",
                "hook line\n",
                HookFile.suffix + "\n",
                "Line 5",
            ],
            True,
        ),
        ([HookFile.suffix + "\n", HookFile.prefix + "\n"], False),
        (
            ["  " + HookFile.prefix + "  \n", "\t" + HookFile.suffix + "\t\n"],
            True,
        ),
        (
            [
                HookFile.prefix + "\n",
                "content",
                HookFile.prefix + "\n",
                HookFile.suffix + "\n",
                HookFile.suffix + "\n",
            ],
            True,
        ),
    ],
    ids=[
        "empty",
        "no_markers",
        "only_prefix",
        "only_suffix",
        "basic_markers",
        "markers_within_content",
        "wrong_order",
        "whitespace_around_markers",
        "multiple_markers",
    ],
)
@patch("vectorcode.subcommands.init.os.path.isfile", return_value=True)
@patch("vectorcode.subcommands.init.open", new_callable=mock_open)
def test_hookfile_has_vectorcode_hooks(
    mock_open_func, mock_isfile, lines, expected, mock_hook_path
):
    """Test HookFile.has_vectorcode_hooks with various line contents."""

    hook_file = HookFile(mock_hook_path)
    hook_file.lines = lines

    assert hook_file.has_vectorcode_hooks() == expected


@patch("vectorcode.subcommands.init.platform.system")
@patch("vectorcode.subcommands.init.os.chmod")
@patch("vectorcode.subcommands.init.os.stat")
@patch("vectorcode.subcommands.init.os.path.isfile")
@patch("vectorcode.subcommands.init.open", new_callable=mock_open)
def test_hookfile_inject_hook_new_file(
    mock_open_func, mock_isfile, mock_stat, mock_chmod, mock_platform, mock_hook_path
):
    """Test injecting hook into a new (non-existent) file."""
    mock_isfile.return_value = False
    mock_platform.return_value = "Linux"

    mock_stat_result = MagicMock()
    mock_stat_result.st_mode = 0o644
    mock_stat.return_value = mock_stat_result

    hook_file = HookFile(mock_hook_path)
    new_content = ["echo 'hello'"]

    hook_file.inject_hook(new_content)

    expected_lines = [
        HookFile.prefix + "\n",
        "echo 'hello'\n",
        HookFile.suffix + "\n",
    ]
    mock_open_func.assert_called_once_with(mock_hook_path, "w")
    handle = mock_open_func()
    handle.writelines.assert_called_once_with(expected_lines)

    mock_stat.assert_called_once_with(mock_hook_path)
    expected_mode = 0o644 | stat.S_IXUSR
    mock_chmod.assert_called_once_with(mock_hook_path, mode=expected_mode)


@patch("vectorcode.subcommands.init.platform.system")
@patch("vectorcode.subcommands.init.os.chmod")
@patch("vectorcode.subcommands.init.os.stat")
@patch("vectorcode.subcommands.init.os.path.isfile")
@patch(
    "vectorcode.subcommands.init.open",
    new_callable=mock_open,
    read_data="Existing line 1\n",
)
def test_hookfile_inject_hook_existing_file_no_vc_hooks(
    mock_open_func, mock_isfile, mock_stat, mock_chmod, mock_platform, mock_hook_path
):
    """Test injecting hook into an existing file without VectorCode hooks."""
    mock_isfile.return_value = True
    mock_platform.return_value = "Windows"

    mock_stat_result = MagicMock()
    mock_stat_result.st_mode = 0o644
    mock_stat.return_value = mock_stat_result

    hook_file = HookFile(mock_hook_path)
    initial_lines = ["Existing line 1\n"]
    assert hook_file.lines == initial_lines

    new_content = ["new hook line 1", "new hook line 2\n"]

    hook_file.inject_hook(new_content)

    expected_lines = initial_lines + [
        HookFile.prefix + "\n",
        "new hook line 1\n",
        "new hook line 2\n",
        HookFile.suffix + "\n",
    ]

    assert mock_open_func.call_count == 2
    mock_open_func.assert_any_call(mock_hook_path)
    mock_open_func.assert_any_call(mock_hook_path, "w")

    handle = mock_open_func()
    handle.writelines.assert_called_once_with(expected_lines)

    mock_stat.assert_not_called()
    mock_chmod.assert_not_called()


@patch("vectorcode.subcommands.init.platform.system")
@patch("vectorcode.subcommands.init.os.chmod")
@patch("vectorcode.subcommands.init.os.stat")
@patch("vectorcode.subcommands.init.os.path.isfile")
@patch("vectorcode.subcommands.init.open", new_callable=mock_open)
def test_hookfile_inject_hook_existing_file_with_vc_hooks(
    mock_open_func, mock_isfile, mock_stat, mock_chmod, mock_platform, mock_hook_path
):
    """Test injecting hook into an existing file that ALREADY has VectorCode hooks."""
    initial_content = [
        "Some line\n",
        HookFile.prefix + "\n",
        "existing hook content\n",
        HookFile.suffix + "\n",
        "Another line\n",
    ]

    read_handle_mock = mock_open(read_data="".join(initial_content)).return_value
    write_handle_mock = mock_open().return_value

    mock_open_func.side_effect = [
        read_handle_mock,
        write_handle_mock,
    ]

    mock_isfile.return_value = True
    mock_platform.return_value = "Linux"

    mock_stat_result = MagicMock()
    mock_stat_result.st_mode = 0o755
    mock_stat.return_value = mock_stat_result

    hook_file = HookFile(mock_hook_path)
    assert hook_file.lines == initial_content

    new_content = ["this should not be added"]
    hook_file.inject_hook(new_content)

    assert hook_file.has_vectorcode_hooks() is True

    assert mock_open_func.call_count == 2
    mock_open_func.assert_has_calls(
        [
            call(mock_hook_path),
            call(mock_hook_path, "w"),
        ]
    )

    write_handle_mock.writelines.assert_called_once_with(initial_content)

    mock_stat.assert_called_once_with(mock_hook_path)
    expected_mode = 0o755 | stat.S_IXUSR
    mock_chmod.assert_called_once_with(mock_hook_path, mode=expected_mode)


@pytest.mark.asyncio
@patch("vectorcode.subcommands.init.find_project_root", return_value=None)
@patch("vectorcode.subcommands.init.load_hooks")
async def test_hooks_orchestration_no_git_repo(mock_load_hooks, mock_find_project_root):
    """Test hooks orchestration: handles no git repo found."""

    with tempfile.TemporaryDirectory() as temp_dir:
        proj_root = os.path.join(temp_dir, "path")
        mock_config = Config(project_root=proj_root, hooks=True)

        return_code = await init(mock_config)

        mock_find_project_root.assert_called_once_with(proj_root, ".git")
        mock_load_hooks.assert_not_called()
        assert return_code == 0


@pytest.mark.asyncio
@patch("vectorcode.subcommands.init.load_hooks")
@patch("vectorcode.subcommands.init.HookFile")
async def test_hooks_orchestration_default_hooks(mock_HookFile, mock_load_hooks):
    """Test hooks orchestration: handles git repo found but no hooks loaded."""
    from vectorcode.subcommands.init import __HOOK_CONTENTS

    __HOOK_CONTENTS.clear()
    __HOOK_CONTENTS.update(
        {
            "pre-commit": [
                "diff_files=$(git diff --cached --name-only)",
                '[ -z "$diff_files" ] || vectorcode vectorise $diff_files',
            ],
            "post-checkout": [
                'files=$(git diff --name-only "$1" "$2")',
                '[ -z "$files" ] || vectorcode vectorise $files',
            ],
        }
    )
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        # patch(
        #     "vectorcode.subcommands.init.find_project_root"
        # ) as mock_find_project_root,
    ):
        project_root = os.path.join(tmp_dir, "fake/project")
        os.makedirs(os.path.join(project_root, ".git/"))
        mock_config = Config(project_root=project_root, hooks=True)

        return_code = await init(mock_config)

        # mock_find_project_root.assert_called_once_with(project_root, ".git")
        mock_load_hooks.assert_called_once()
        mock_HookFile.assert_has_calls(
            [
                call(
                    os.path.join(project_root, ".git/hooks/pre-commit"),
                    git_dir=project_root,
                ),
                call().inject_hook(
                    [
                        "diff_files=$(git diff --cached --name-only)",
                        '[ -z "$diff_files" ] || vectorcode vectorise $diff_files',
                    ],
                    False,
                ),
                call(
                    os.path.join(project_root, ".git/hooks/post-checkout"),
                    git_dir=project_root,
                ),
                call().inject_hook(
                    [
                        'files=$(git diff --name-only "$1" "$2")',
                        '[ -z "$files" ] || vectorcode vectorise $files',
                    ],
                    False,
                ),
            ],
            any_order=True,
        )
        assert return_code == 0


@patch("vectorcode.subcommands.init.os.path.isfile", return_value=True)
@patch(
    "vectorcode.subcommands.init.open",
    new_callable=mock_open,
)
def test_hookfile_has_vectorcode_hooks_force_removes_block(
    mock_open_func, mock_isfile, mock_hook_path
):
    """Test that has_vectorcode_hooks with force=True removes the existing block."""
    initial_lines = [
        "Line 1\n",
        HookFile.prefix + "\n",
        "old hook line\n",
        HookFile.suffix + "\n",
        "Line 5\n",
    ]
    expected_lines_after = [
        "Line 1\n",
        "Line 5\n",
    ]

    # Mock reading the initial content
    mock_open_func.side_effect = [
        mock_open(read_data="".join(initial_lines)).return_value
    ]

    hook_file = HookFile(mock_hook_path)
    assert hook_file.lines == initial_lines  # Ensure lines were read

    # Call with force=True
    found = hook_file.has_vectorcode_hooks(force=True)

    assert found is False  # Should return False because it modifies in place
    assert hook_file.lines == expected_lines_after  # Check if block was removed


@patch("vectorcode.subcommands.init.platform.system")
@patch("vectorcode.subcommands.init.os.chmod")
@patch("vectorcode.subcommands.init.os.stat")
@patch("vectorcode.subcommands.init.os.path.isfile")
@patch("vectorcode.subcommands.init.open", new_callable=mock_open)
def test_hookfile_inject_hook_force_overwrites_existing(
    mock_open_func, mock_isfile, mock_stat, mock_chmod, mock_platform, mock_hook_path
):
    """Test inject_hook with force=True correctly overwrites an existing hook block."""
    initial_content = [
        "Some line\n",
        f"  {HookFile.prefix}  \n",  # With whitespace
        "existing hook content\n",
        f"\t{HookFile.suffix}\t\n",  # With whitespace
        "Another line\n",
    ]
    new_hook_content = ["new hook line 1", "new hook line 2\n"]

    # Mock the sequence: read initial, then write final
    read_handle_mock = mock_open(read_data="".join(initial_content)).return_value
    write_handle_mock = mock_open().return_value
    mock_open_func.side_effect = [
        read_handle_mock,  # Initial read in HookFile.__init__
        write_handle_mock,  # Write in inject_hook
    ]

    mock_isfile.return_value = True
    mock_platform.return_value = "Linux"  # To trigger chmod

    mock_stat_result = MagicMock()
    mock_stat_result.st_mode = 0o644
    mock_stat.return_value = mock_stat_result

    hook_file = HookFile(mock_hook_path)
    assert hook_file.lines == initial_content  # Verify initial state

    # Inject with force=True
    hook_file.inject_hook(new_hook_content, force=True)

    # Verify the final content written to the file
    expected_lines_written = [
        "Some line\n",
        "Another line\n",  # Existing block removed
        HookFile.prefix + "\n",  # New block added
        "new hook line 1\n",
        "new hook line 2\n",
        HookFile.suffix + "\n",
    ]

    # Check calls
    assert mock_open_func.call_count == 2  # Read + Write
    mock_open_func.assert_has_calls(
        [
            call(mock_hook_path),  # Initial read
            call(mock_hook_path, "w"),  # Write
        ]
    )
    write_handle_mock.writelines.assert_called_once_with(expected_lines_written)

    # Check permissions were set
    mock_stat.assert_called_once_with(mock_hook_path)
    expected_mode = 0o644 | stat.S_IXUSR
    mock_chmod.assert_called_once_with(mock_hook_path, mode=expected_mode)


@pytest.mark.asyncio
@patch("vectorcode.subcommands.init.load_hooks")
@patch("vectorcode.subcommands.init.HookFile")
async def test_hooks_orchestration_force_true(mock_HookFile, mock_load_hooks):
    """Test hooks orchestration passes force=True to HookFile.inject_hook."""
    from vectorcode.subcommands.init import __HOOK_CONTENTS

    # Ensure there's some hook content defined for the test
    defined_hooks = {"pre-commit": ["echo pre-commit"]}
    __HOOK_CONTENTS.clear()
    __HOOK_CONTENTS.update(defined_hooks)

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = os.path.join(tmpdir, "project")
        mock_config = Config(
            project_root=project_root, force=True, hooks=True
        )  # Set force=True

        os.makedirs(os.path.join(project_root, ".git"))
        # Mock the HookFile instance and its methods
        mock_hook_instance = MagicMock()
        mock_HookFile.return_value = mock_hook_instance

        return_code = await init(mock_config)

        # Assertions
        mock_load_hooks.assert_called_once()

        # Check HookFile was instantiated correctly
        expected_hook_path = os.path.join(project_root, ".git/hooks/pre-commit")
        mock_HookFile.assert_called_once_with(expected_hook_path, git_dir=project_root)

        # Crucially, check inject_hook was called with force=True
        mock_hook_instance.inject_hook.assert_called_once_with(
            defined_hooks["pre-commit"],
            True,  # force=True passed here
        )

        assert return_code == 0
