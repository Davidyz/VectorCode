import pytest

from vectorcode.cli_utils import GLOBAL_CONFIG_DIR


@pytest.fixture(autouse=True)
def restore_global_config_path():
    global GLOBAL_CONFIG_DIR
    original_global_config_path = GLOBAL_CONFIG_DIR
    yield
    GLOBAL_CONFIG_DIR = original_global_config_path
