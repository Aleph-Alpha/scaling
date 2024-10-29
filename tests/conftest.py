import os
from pathlib import Path

import pytest
from _pytest.config import Config
from _pytest.python import Function


def pytest_collection_modifyitems(items: list[Function], config: Config) -> None:
    default_markers = {"filterwarnings", "skip", "skipif", "xfail", "parametrize", "usefixtures", "tryfirst", "trylast"}
    names_list = set()
    for line in config.getini("markers"):
        names, rest = line.split(":", 1)
        if "(" in names:
            names, rest = names.split("(", 1)

        names_list.add(names)

    names_list -= default_markers

    for item in items:
        if not any([marker.name in names_list for marker in item.iter_markers()]):
            item.add_marker("unmarked")


@pytest.fixture(autouse=True, scope="function")
def reset_test_state() -> None:
    if "DET_LATEST_CHECKPOINT" in os.environ:
        del os.environ["DET_LATEST_CHECKPOINT"]


@pytest.fixture(scope="session")
def path_to_root():
    return Path(__file__).parents[0].parents[0]
