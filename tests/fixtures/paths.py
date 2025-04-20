from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_path() -> Path:
    return Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def root_path(fixtures_path: Path) -> Path:
    return fixtures_path.parent


@pytest.fixture(scope="session")
def data_path(root_path: Path) -> Path:
    return root_path / "data"


# @pytest.fixture(scope="session")
# def cache_path(root_path: Path) -> Path:
#     return root_path / "cache"


@pytest.fixture(scope="session")
def cache_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="session")
def tests_path(root_path: Path) -> Path:
    return root_path / "tests"
