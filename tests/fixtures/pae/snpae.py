from typing import TYPE_CHECKING

import pytest

import suPAErnova

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.pae import PAEStep
    from suPAErnova.steps.pae.model import PAEStepResult


@pytest.fixture(scope="session")
def snpae_pae_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "Callable[[dict[str, Any], dict[str, Any]], PAEStep]":
    def _snpae_pae_step(
        data_params: "dict[str, Any]", pae_params: "dict[str, Any]"
    ) -> "PAEStep":
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
            },
            "pae": {
                "model": pae_params["model"],
                "validation_frac": pae_params["validation_frac"],
            },
        }
        snpae = suPAErnova.prepare_config(
            config,
            base_path=root_path,
            out_path=cache_path / pae_params["fname"] / "pae" / "snpae",
        )
        snpae.run()
        paestep = snpae.pae_step
        assert paestep is not None, "Error running PAEStep"
        return paestep

    return _snpae_pae_step


@pytest.fixture(scope="session")
def snpae_pae_result_factory(
    snpae_pae_step_factory: "Callable[[dict[str, Any], dict[str, Any]], PAEStep]",
) -> "Callable[[dict[str, Any], dict[str, Any]], list[PAEStepResult]]":
    def _snpae_pae_result(
        data_params: dict[str, "Any"], pae_params: "dict[str, Any]"
    ) -> "list[PAEStepResult]":
        return snpae_pae_step_factory(data_params, pae_params).pae

    return _snpae_pae_result
