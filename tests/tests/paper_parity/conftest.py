from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.configs.steps.pae import PAEStepResult
    from suPAErnova.configs.steps.data import DataStepResult

pytestmark = pytest.mark.paper_parity

# --- Data Step ---


@pytest.fixture(scope="module")
def data_params() -> dict[str, "Any"]:
    return {
        "min_phase": -10,
        "max_phase": 40,
        "train_frac": 0.75,
        "seed": 12345,
        "fname": "paper_parity",
    }


@pytest.fixture(scope="module")
def snpae_data(
    data_params: dict[str, "Any"],
    snpae_data_result_factory: "Callable[[dict[str, Any]], DataStepResult]",
) -> "DataStepResult":
    return snpae_data_result_factory(data_params)


@pytest.fixture(scope="module")
def legacy_data(
    data_params: dict[str, "Any"],
    legacy_data_result_factory: "Callable[[dict[str, Any]], DataStepResult]",
) -> "DataStepResult":
    return legacy_data_result_factory(data_params)


# --- PAE Step ---


@pytest.fixture(scope="module")
def pae_params(data_path: "Path") -> dict[str, "Any"]:
    return {"fname": "paper_parity"}


@pytest.fixture(scope="module")
def snpae_pae(
    data_params: dict[str, "Any"],
    pae_params: dict[str, "Any"],
    snpae_pae_result_factory: "Callable[[dict[str, Any], dict[str, Any]], PAEStepResult]",
) -> "PAEStepResult":
    return snpae_pae_result_factory(data_params, pae_params)


@pytest.fixture(scope="module")
def legacy_pae(
    data_params: dict[str, "Any"],
    pae_params: dict[str, "Any"],
    legacy_pae_result_factory: "Callable[[dict[str, Any], dict[str, Any]], PAEStepResult]",
) -> "PAEStepResult":
    return legacy_pae_result_factory(data_params, pae_params)
