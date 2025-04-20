from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    from suPAErnova.steps.data import SNPAEData


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
    snpae_data_result_factory: "Callable[[dict[str, Any]], SNPAEData]",
) -> "SNPAEData":
    return snpae_data_result_factory(data_params)


@pytest.fixture(scope="module")
def legacy_data(
    data_params: dict[str, "Any"],
    legacy_data_result_factory: "Callable[[dict[str, Any]], SNPAEData]",
) -> "SNPAEData":
    return legacy_data_result_factory(data_params)
