from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.pae import PAEStep
    from suPAErnova.steps.pae.model import PAEStepResult


@pytest.fixture(scope="session")
def legacy_pae_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "Callable[[dict[str, Any], dict[str, Any]], PAEStep]":
    def _legacy_pae_step(
        data_params: "dict[str, Any]", pae_params: "dict[str, Any]"
    ) -> "PAEStep":
        pass

    return _legacy_pae_step


@pytest.fixture(scope="session")
def legacy_pae_result_factory(
    legacy_pae_step_factory: "Callable[[dict[str, Any], dict[str, Any]], PAEStep]",
) -> "Callable[[dict[str, Any], dict[str, Any]], list[PAEStepResult]]":
    def _legacy_pae_result(
        data_params: dict[str, "Any"], pae_params: "dict[str, Any]"
    ) -> "list[PAEStepResult]":
        return legacy_pae_step_factory(data_params, pae_params).pae

    return _legacy_pae_result
