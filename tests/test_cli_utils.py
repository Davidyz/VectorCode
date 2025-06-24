import os
import subprocess
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import pytest

from vectorcode import cli_utils
from vectorcode.cli_utils import (
    CliAction,
    Config,
    PromptCategory,
    QueryInclude,
    cleanup_path,
    expand_envs_in_dict,
    expand_globs,
    expand_path,
    find_project_config_dir,
    find_project_root,
    get_project_config,
    load_config_file,
    parse_cli_args,
)


@pytest.mark.asyncio
async def test_config_import_from():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        os.makedirs(db_path, exist_ok=True)
        config_dict: Dict[str, Any] = {
            "db_path": db_path,
            "db_url": "http://test_host:1234",
            "embedding_function": "TestEmbedding",
            "embedding_params": {"param1": "value1"},
            "chunk_size": 512,
            "overlap_ratio": 0.3,
            "query_multiplier": 5,
            "reranker": "TestReranker",
            "reranker_params": {"reranker_param1": "reranker_value1"},
            "db_settings": {"db_setting1": "db_value1"},
        }
        config = await Config.import_from(config_dict)
        assert config.db_path == db_path
        assert config.db_log_path == os.path.expanduser("~/.local/share/vectorcode/")
        assert config.db_url == "http://test_host:1234"
        assert config.embedding_function == "TestEmbedding"
        assert config.embedding_params == {"param1": "value1"}
        assert config.chunk_size == 512
        assert config.overlap_ratio == 0.3
        assert config.query_multiplier == 5
        assert config.reranker == "TestReranker"
        assert config.reranker_params == {"reranker_param1": "reranker_value1"}
        assert config.db_settings == {"db_setting1": "db_value1"}


@pytest.mark.asyncio
async def test_config_import_from_invalid_path():
    config_dict: Dict[str, Any] = {"db_path": "/path/does/not/exist"}
    with pytest.raises(IOError):
        await Config.import_from(config_dict)


@pytest.mark.asyncio
async def test_config_import_from_db_path_is_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db_file")
        with open(db_path, "w") as f:
            f.write("test")

        config_dict: Dict[str, Any] = {"db_path": db_path}
        with pytest.raises(IOError):
            await Config.import_from(config_dict)


@pytest.mark.asyncio
async def test_config_merge_from():
    config1 = Config(db_url="http://host1:8001", n_result=5)
    config2 = Config(db_url="http://host2:8002", query=["test"])
    merged_config = await config1.merge_from(config2)
    assert merged_config.db_url == "http://host2:8002"
    assert merged_config.n_result == 5
    assert merged_config.query == ["test"]


@pytest.mark.asyncio
async def test_config_merge_from_new_fields():
    config1 = Config(db_url="http://host1:8001")
    config2 = Config(query=["test"], n_result=10, recursive=True)
    merged_config = await config1.merge_from(config2)
    assert merged_config.db_url == "http://host1:8001"
    assert merged_config.query == ["test"]
    assert merged_config.n_result == 10
    assert merged_config.recursive


@pytest.mark.asyncio
async def test_config_import_from_missing_keys():
    config_dict: Dict[str, Any] = {}  # Empty dictionary, all keys missing
    config = await Config.import_from(config_dict)

    # Assert that default values are used
    assert config.embedding_function == "SentenceTransformerEmbeddingFunction"
    assert config.embedding_params == {}
    assert config.db_url == "http://127.0.0.1:8000"
    assert config.db_path == os.path.expanduser("~/.local/share/vectorcode/chromadb/")
    assert config.chunk_size == 2500
    assert config.overlap_ratio == 0.2
    assert config.query_multiplier == -1
    assert config.reranker == "CrossEncoderReranker"
    assert config.reranker_params == {}
    assert config.db_settings is None


def test_expand_envs_in_dict():
    os.environ["TEST_VAR"] = "test_value"
    d = {"key1": "$TEST_VAR", "key2": {"nested_key": "$TEST_VAR"}}
    expand_envs_in_dict(d)
    assert d["key1"] == "test_value"
    assert d["key2"]["nested_key"] == "test_value"

    d = {"key3": "$NON_EXISTENT_VAR"}
    expand_envs_in_dict(d)
    assert d["key3"] == "$NON_EXISTENT_VAR"  # Should remain unchanged

    d = {"key4": "$TEST_VAR2"}
    expand_envs_in_dict(d)
    assert d["key4"] == "$TEST_VAR2"  # Should remain unchanged

    del os.environ["TEST_VAR"]  # Clean up the env


@pytest.mark.asyncio
async def test_expand_globs():
    with tempfile.TemporaryDirectory() as temp_dir:
        file1_path = os.path.join(temp_dir, "file1.txt")
        dir1_path = os.path.join(temp_dir, "dir1")
        file2_path = os.path.join(dir1_path, "file2.txt")
        dir2_path = os.path.join(temp_dir, "dir2")
        file3_path = os.path.join(dir2_path, "file3.txt")

        os.makedirs(dir1_path, exist_ok=True)
        os.makedirs(dir2_path, exist_ok=True)

        with open(file1_path, "w") as f:
            f.write("content")
        with open(file2_path, "w") as f:
            f.write("content")
        with open(file3_path, "w") as f:
            f.write("content")

        paths = [file1_path, dir1_path]
        expanded_paths = await expand_globs(paths, recursive=True)
        assert len(expanded_paths) == 2
        assert file1_path in expanded_paths
        assert file2_path in expanded_paths

        paths = [os.path.join(temp_dir, "*.txt")]
        expanded_paths = await expand_globs(paths, recursive=False)
        assert len(expanded_paths) == 1  # Expecting 1 file in the temp_dir

        paths = [dir1_path]
        expanded_paths = await expand_globs(paths, recursive=True)
        assert len(expanded_paths) == 1
        assert file2_path in expanded_paths

        assert len(await expand_globs([os.path.join(temp_dir, "**", "*.txt")])) == 3


def test_expand_path():
    path_with_user = "~/test_dir"
    expanded_path = expand_path(path_with_user)
    assert expanded_path == os.path.join(os.path.expanduser("~"), "test_dir")

    os.environ["TEST_VAR"] = "test_value"
    path_with_env = "$TEST_VAR/test_dir"
    expanded_path = expand_path(path_with_env)
    assert expanded_path == os.path.join("test_value", "test_dir")

    abs_path = "/tmp/test_dir"
    expanded_path = expand_path(abs_path)
    assert expanded_path == abs_path

    rel_path = "test_dir"
    expanded_path = expand_path(rel_path)
    assert expanded_path == rel_path

    abs_path = "~/test_dir"
    expanded_path = expand_path(abs_path, absolute=True)
    assert expanded_path == os.path.abspath(os.path.expanduser(abs_path))


@pytest.mark.asyncio
async def test_load_config_file_invalid_json():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            f.write("invalid json")

        with pytest.raises(ValueError):
            await load_config_file(config_path)


@pytest.mark.asyncio
async def test_load_from_default_config():
    for name in ("config.json5", "config.json"):
        with (
            tempfile.TemporaryDirectory() as fake_home,
        ):
            os.environ.update({"HOME": fake_home})
            config_path = os.path.join(fake_home, ".config", "vectorcode", name)
            config_dir = os.path.join(fake_home, ".config", "vectorcode")
            setattr(
                cli_utils,
                "GLOBAL_CONFIG_DIR",
                config_dir,
            )
            os.makedirs(config_dir, exist_ok=True)
            config_content = '{"db_url": "http://default.url:8000"}'
            with open(config_path, "w") as fin:
                fin.write(config_content)

            config = await load_config_file()
            assert config.db_url == "http://default.url:8000"


@pytest.mark.asyncio
async def test_load_config_file_invalid_config():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('"hello world"')

        with pytest.raises(ValueError):
            await load_config_file(config_path)


@pytest.mark.asyncio
async def test_find_project_config_dir_no_anchors():
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = await find_project_config_dir(temp_dir)
        assert project_dir is None


@pytest.mark.asyncio
async def test_expand_globs_nonexistent_path():
    expanded_paths = await expand_globs(["/path/does/not/exist"])
    assert len(expanded_paths) == 0


@pytest.mark.asyncio
async def test_load_config_file_empty_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            f.write("")

        assert await load_config_file(config_path) == Config()


@pytest.mark.asyncio
async def test_find_project_config_dir_nested():
    with tempfile.TemporaryDirectory() as temp_dir:
        level1_dir = os.path.join(temp_dir, "level1")
        level2_dir = os.path.join(level1_dir, "level2")
        level3_dir = os.path.join(level2_dir, "level3")
        os.makedirs(level3_dir)

        # Create a .vectorcode directory in level2
        vectorcode_dir = os.path.join(level2_dir, ".vectorcode")
        os.makedirs(vectorcode_dir)

        # Create a .git directory in level1
        git_dir = os.path.join(level1_dir, ".git")
        os.makedirs(git_dir)

        # Test finding from level3_dir; should find .vectorcode in level2
        found_dir = await find_project_config_dir(level3_dir)
        assert found_dir is not None and os.path.samefile(found_dir, vectorcode_dir)

        # Test finding from level2_dir; should find .vectorcode in level2
        found_dir = await find_project_config_dir(level2_dir)
        assert found_dir is not None and os.path.samefile(found_dir, vectorcode_dir)

        # Test finding from level1_dir; should find .git in level1
        found_dir = await find_project_config_dir(level1_dir)
        assert found_dir is not None and os.path.samefile(found_dir, git_dir)


@pytest.mark.asyncio
async def test_expand_globs_mixed_paths():
    with tempfile.TemporaryDirectory() as temp_dir:
        existing_file = os.path.join(temp_dir, "existing_file.txt")
        with open(existing_file, "w") as f:
            f.write("content")

        paths = [existing_file, "/path/does/not/exist"]
        expanded_paths = await expand_globs(paths)
        assert len(expanded_paths) == 1
        assert existing_file in expanded_paths


@pytest.mark.asyncio
async def test_cli_arg_parser():
    with patch(
        "sys.argv", ["vectorcode", "query", "test_query", "-n", "5", "--absolute"]
    ):
        config = await parse_cli_args()
        assert config.action == CliAction.query
        assert config.query == ["test_query"]
        assert config.n_result == 5
        assert config.use_absolute_path


def test_query_include_to_header():
    assert QueryInclude.path.to_header() == "Path: "
    assert QueryInclude.document.to_header() == "Document:\n"


def test_find_project_root():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test when no anchor file exists
        assert find_project_root(temp_dir) is None

        # Test when anchor file exists
        os.makedirs(os.path.join(temp_dir, ".vectorcode"))
        found_dir = find_project_root(temp_dir)
        assert found_dir is not None and os.path.samefile(found_dir, temp_dir)

        # Test when start_from is a file
        test_file = os.path.join(temp_dir, "test_file.txt")
        open(test_file, "a").close()
        found_dir = find_project_root(test_file)
        assert found_dir is not None and os.path.samefile(found_dir, temp_dir)


@pytest.mark.asyncio
async def test_get_project_config_no_local_config():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = await get_project_config(temp_dir)
        assert config.chunk_size == Config().chunk_size, "Should load default value."


@pytest.mark.asyncio
async def test_parse_cli_args_no_action():
    with patch("sys.argv", ["vectorcode"]):
        with pytest.raises(SystemExit):
            await parse_cli_args()


@pytest.mark.asyncio
async def test_parse_cli_args_include_argument():
    with patch(
        "sys.argv",
        [
            "vectorcode",
            "query",
            "test_query",
            "--include",
            "path",
            "document",
        ],
    ):
        config = await parse_cli_args()
        assert config.include == [QueryInclude.path, QueryInclude.document]


@pytest.mark.asyncio
async def test_parse_cli_args_vectorise():
    with patch("sys.argv", ["vectorcode", "vectorise", "file1.txt"]):
        config = await parse_cli_args()
        assert config.action == CliAction.vectorise
        assert config.files == ["file1.txt"]


@pytest.mark.asyncio
async def test_parse_cli_args_vectorise_encoding():
    with patch(
        "sys.argv", ["vectorcode", "vectorise", "file1.txt", "--encoding", "gbk"]
    ):
        config = await parse_cli_args()
        assert config.action == CliAction.vectorise
        assert config.files == ["file1.txt"]
        assert config.encoding == "gbk"


@pytest.mark.asyncio
async def test_parse_cli_args_vectorise_recursive_dir():
    with patch("sys.argv", ["vectorcode", "vectorise", "-r", "."]):
        config = await parse_cli_args()
        assert config.action == CliAction.vectorise
        assert config.files == ["."]
        assert config.recursive is True
        assert config.include_hidden is False


@pytest.mark.asyncio
async def test_parse_cli_args_vectorise_recursive_dir_include_hidden():
    with patch("sys.argv", ["vectorcode", "vectorise", "-r", "."]):
        config = await parse_cli_args()
        assert config.action == CliAction.vectorise
        assert config.files == ["."]
        assert config.recursive is True
        assert config.include_hidden is False


@pytest.mark.asyncio
async def test_parse_cli_args_vectorise_no_files():
    with patch("sys.argv", ["vectorcode", "vectorise"]):
        config = await parse_cli_args()
        assert config.action == CliAction.vectorise
        assert config.files == []


@pytest.mark.asyncio
async def test_get_project_config_local_config(tmp_path):
    project_root = tmp_path / "project"
    vectorcode_dir = project_root / ".vectorcode"
    vectorcode_dir.mkdir(parents=True)

    config_file = vectorcode_dir / "config.json"
    config_file.write_text('{"db_url": "http://test_host:9999" }')

    config = await get_project_config(project_root)
    assert config.db_url == "http://test_host:9999"


@pytest.mark.asyncio
async def test_get_project_config_local_config_json5(tmp_path):
    project_root = tmp_path / "project"
    vectorcode_dir = project_root / ".vectorcode"
    vectorcode_dir.mkdir(parents=True)

    config_file = vectorcode_dir / "config.json5"
    config_file.write_text('{"db_url": "http://test_host:9999" }')

    config = await get_project_config(project_root)
    assert config.db_url == "http://test_host:9999"


def test_find_project_root_file_input(tmp_path):
    # Create a temporary file
    temp_file = tmp_path / "temp_file.txt"
    temp_file.write_text("test content")

    # Create a .vectorcode directory in the parent directory
    vectorcode_dir = tmp_path / ".vectorcode"
    vectorcode_dir.mkdir()

    # Call find_project_root with the temporary file path
    project_root = find_project_root(temp_file)

    # Assert that it returns the parent directory (tmp_path)
    assert os.path.samefile(str(project_root), str(tmp_path))


@pytest.mark.asyncio
async def test_parse_cli_args_update():
    with patch("sys.argv", ["vectorcode", "update"]):
        config = await parse_cli_args()
        assert config.action == CliAction.update


@pytest.mark.asyncio
async def test_parse_cli_args_clean():
    with patch("sys.argv", ["vectorcode", "clean"]):
        config = await parse_cli_args()
        assert config.action == CliAction.clean


@pytest.mark.asyncio
async def test_parse_cli_args_check():
    with patch("sys.argv", ["vectorcode", "check", "config"]):
        config = await parse_cli_args()
        assert config.action == CliAction.check
        assert config.check_item == "config"


@pytest.mark.asyncio
async def test_parse_cli_args_init():
    with patch("sys.argv", ["vectorcode", "init"]):
        config = await parse_cli_args()
        assert config.action == CliAction.init


@pytest.mark.asyncio
async def test_parse_cli_args_prompts():
    with patch("sys.argv", ["vectorcode", "prompts", "ls"]):
        config = await parse_cli_args()
        assert config.action == CliAction.prompts
        assert config.prompt_categories == [PromptCategory.ls]


@pytest.mark.asyncio
async def test_parse_cli_args_chunks():
    with patch(
        "sys.argv", ["vectorcode", "chunks", "file.py", "-c", "100", "-o", "0.5"]
    ):
        config = await parse_cli_args()
        assert config.action == CliAction.chunks
        assert config.overlap_ratio == 0.5
        assert config.chunk_size == 100

    with patch("sys.argv", ["vectorcode", "chunks", "file.py"]):
        config = await parse_cli_args()
        assert config.action == CliAction.chunks
        assert config.overlap_ratio == Config().overlap_ratio
        assert config.chunk_size == Config().chunk_size


@pytest.mark.asyncio
async def test_config_import_from_hnsw():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        os.makedirs(db_path, exist_ok=True)
        config_dict: Dict[str, Any] = {
            "hnsw": {"space": "cosine", "ef_construction": 200, "m": 32}
        }
        config = await Config.import_from(config_dict)
        assert config.hnsw["space"] == "cosine"
        assert config.hnsw["ef_construction"] == 200
        assert config.hnsw["m"] == 32


@pytest.mark.asyncio
async def test_hnsw_config_merge():
    config1 = Config(hnsw={"space": "ip"})
    config2 = Config(hnsw={"ef_construction": 200})
    merged_config = await config1.merge_from(config2)
    assert merged_config.hnsw["space"] == "ip"
    assert merged_config.hnsw["ef_construction"] == 200


def test_cleanup_path():
    home = os.environ.get("HOME")
    if home is None:
        return
    assert cleanup_path(os.path.join(home, "test_path")) == os.path.join(
        "~", "test_path"
    )
    assert cleanup_path("/etc/dir") == "/etc/dir"


def test_shtab():
    for shell in ("bash", "zsh", "tcsh"):
        assert (
            subprocess.Popen(
                [sys.executable, "-m", "vectorcode.main", "-s", shell],
                stderr=subprocess.PIPE,
            )
            .stderr.read()
            .decode()
        ) == ""
