from typing import TYPE_CHECKING, Literal

import pytest

import suPAErnova

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.pae import PAEStep
    from suPAErnova.configs.steps.pae import PAEStepResult

    PAE = PAEStep[Literal["tf"]]


@pytest.fixture(scope="session")
def snpae_pae_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "Callable[[dict[str, Any]], PAE]":
    def _snpae_pae_step(pae_params: "dict[str, Any]") -> "PAE":
        config: "dict[str, Any]" = {}
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
    snpae_pae_step_factory: "Callable[[dict[str, Any]], PAE]",
) -> "Callable[[dict[str, Any]], list[PAEStepResult]]":
    def _snpae_pae_result(pae_params: dict[str, "Any"]) -> "list[PAEStepResult]":
        return snpae_pae_step_factory(pae_params).results

    return _snpae_pae_result
