from typing import TYPE_CHECKING

import pytest

import suPAErnova

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.steps.data import DataStepResult


@pytest.fixture(scope="session")
def snpae_data_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "Callable[[dict[str, Any]], DataStep]":
    def _snpae_data_step(data_params: "dict[str, Any]") -> "DataStep":
        config: "dict[str, Any]" = {
            "data": {
                "data_dir": data_path,
                "meta": "meta.csv",
                "idr": "IDR_eTmax.txt",
                "mask": "mask_info_wmin_wmax.txt",
                "min_phase": data_params["min_phase"],
                "max_phase": data_params["max_phase"],
                "train_frac": data_params["train_frac"],
                "seed": data_params["seed"],
            }
        }
        snpae = suPAErnova.prepare_config(
            config,
            base_path=root_path,
            out_path=cache_path / data_params["fname"] / "data" / "snpae",
        )
        snpae.run()
        datastep = snpae.data_step
        assert datastep is not None, "Error running DataStep"
        return datastep

    return _snpae_data_step


@pytest.fixture(scope="session")
def snpae_data_result_factory(
    snpae_data_step_factory: "Callable[[dict[str, Any]], DataStep]",
) -> "Callable[[dict[str, Any]], DataStepResult]":
    def _snpae_data_result(data_params: dict[str, "Any"]) -> "DataStepResult":
        return snpae_data_step_factory(data_params).data

    return _snpae_data_result
