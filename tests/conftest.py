from typing import TYPE_CHECKING

import pytest

from .fixtures.paths import data_path, root_path, cache_path, tests_path, fixtures_path
from .fixtures.data.snpae import snpae_data_step_factory, snpae_data_result_factory
from .fixtures.data.legacy import legacy_data_step_factory, legacy_data_result_factory

if TYPE_CHECKING:
    from _pytest.config import Config


def pytest_configure(config: "Config") -> None:
    config.addinivalue_line(
        "markers",
        'setup("snpae"|"legacy_snpae"): Setup cached data without running any tests',
    )

    config.addinivalue_line(
        "markers",
        "paper_parity: Paper parity tests",
    )

    config.addinivalue_line(
        "markers",
        "data: Data step tests",
    )


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--setup",
        action="store",
        help='--setup "snpae"|"legacy_snpae": Setup cached data without running any tests',
    )


def pytest_runtest_setup(item) -> None:
    setup = [mark.args[0] for mark in item.iter_markers(name="setup")]
    if setup:
        if item.config.getoption("--setup") not in setup:
            pytest.skip("Setup skipped")
    elif item.config.getoption("--setup"):
        pytest.skip("Setting up, so not running tests")
