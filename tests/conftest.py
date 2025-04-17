import os
from typing import TYPE_CHECKING
from pathlib import Path

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.python import Function
    from _pytest.fixtures import SubRequest
    from _pytest.config.argparsing import Parser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

pytest_plugins = ["tests.test_parity.fixtures.data_step"]


@pytest.fixture(scope="session")
def root_path() -> Path:
    return Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def data_path(root_path: Path) -> Path:
    return root_path / "suPAErnova_data"


@pytest.fixture(scope="session")
def cache_path(root_path: Path) -> Path:
    return root_path / "cache"


@pytest.fixture(scope="session")
def force(request: "SubRequest"):
    return request.config.getoption("--force")


@pytest.fixture(scope="session")
def verbosity(request: "SubRequest"):
    return request.config.getoption("verbose")


def pytest_addoption(parser: "Parser") -> None:
    parser.addoption("--force", action="store_true")


def pytest_collection_modifyitems(config: "Config", items: list["Function"]) -> None:
    keywordexpr = config.option.keyword
    markexpr = config.option.markexpr
    if keywordexpr or markexpr:
        return  # let pytest handle this

    skip_setup = pytest.mark.skip(reason="setup marker not selected")
    for item in items:
        if "setup" in item.keywords:
            item.add_marker(skip_setup)
