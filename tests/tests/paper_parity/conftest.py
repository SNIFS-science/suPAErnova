from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.data import DataStepResult
    from suPAErnova.steps.pae.model import PAEStepResult

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
    return {
        "model": {
            "architecture": "dense",
            "encode_dims": [256, 128],
            "n_z_latents": 3,
            "physical_latents": True,
            "batch_normalisation": True,
            "dropout": 0.2,
            "backend": "tf",
            "seperate_z_latent_training": True,
            "seperate_latent_training": True,
            "colourlaw": data_path / "colourlaws" / "F99_colourlaw.txt",
            "loss_residual_penalty": 0.0,
            "loss_delta_m_penalty": 0.0,
            "loss_covariance_penalty": 0.0,
            "loss_decorrelate_dust": False,
            "loss_decorrelate_all": False,
            "seed": 12345,
            "min_train_redshift": 0.02,
            "max_train_redshift": 1.0,
            "min_train_phase": -10,
            "max_train_phase": 40,
            "activation": "relu",
            "kernel_regulariser": "l2",
            "kernel_regulariser_penalty": 0.01,
            "scheduler": "ExponentialDecay",
            "optimiser": "AdamW",
            "loss": "Huber",
            "amplitude_offset_scale": 1.0,
            "mask_fraction": 0.1,
            "phase_offset_scale": -0.02,
            "debug": False,
        },
        "validation_frac": 0.33,
        "fname": "paper_parity",
    }


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
