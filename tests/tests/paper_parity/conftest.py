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
        "cosmological_model": "WMAP7",
        "salt_model": "salt2",
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
def pae_params(
    data_path: "Path",
) -> dict[str, "Any"]:
    return {
        "validation_frac": 0,
        "save_best": False,
        "delta_av_epochs": 1000,
        "zs_epochs": 1000,
        "delta_m_epochs": 5000,
        "delta_p_epochs": 5000,
        "final_epochs": 5000,
        "delta_av_lr": 0.005,
        "zs_lr": 0.005,
        "delta_m_lr": 0.005,
        "delta_p_lr": 0.001,
        "final_lr": 0.001,
        "delta_av_lr_decay_steps": 300,
        "zs_lr_decay_steps": 300,
        "delta_m_lr_decay_steps": 300,
        "delta_p_lr_decay_steps": 300,
        "final_lr_decay_steps": 300,
        "delta_av_lr_decay_rate": 0.95,
        "zs_lr_decay_rate": 0.95,
        "delta_m_lr_decay_rate": 0.95,
        "delta_p_lr_decay_rate": 0.95,
        "final_lr_decay_rate": 0.95,
        "delta_av_lr_weight_decay_rate": 0.0001,
        "zs_lr_weight_decay_rate": 0.0001,
        "delta_m_lr_weight_decay_rate": 0.0001,
        "delta_p_lr_weight_decay_rate": 0.0001,
        "final_lr_weight_decay_rate": 0.0001,
        "fname": "paper_parity",
        "min_train_redshift": 0.02,
        "min_test_redshift": 0.02,
        "min_val_redshift": 0.02,
        "max_train_redshift": 1.0,
        "max_test_redshift": 1.0,
        "max_val_redshift": 1.0,
        "min_train_phase": -10,
        "min_test_phase": -10,
        "min_val_phase": -10,
        "max_train_phase": 40,
        "max_test_phase": 40,
        "max_val_phase": 40,
        "n_z_latents": 3,
        "seed": 12345,
        "activation": "relu",
        "loss": "WHuber",
        "optimiser": "AdamW",
        "scheduler": "ExponentialDecay",
        "colourlaw": data_path / "colourlaws" / "F99_colourlaw.txt",
        "kernel_regulariser": None,
        "kernel_regulariser_penalty": 100,  # Not used if kernel_regulariser is None
        "architecture": "dense",
        "encode_dims": (256, 128),
        "dropout": 0,
        "batch_normalisation": False,
        "physical_latents": True,
        "batch_size": 57,  # Only correct for the SNFactory data
        "val_every": 100,
        "amplitude_offset_scale": 1.0,
        "phase_offset_scale": -0.02,
        "mask_fraction": 0.1,
        "loss_clip_delta": 25,
        "loss_residual_penalty": 0,
        "use_amplitude": True,
        "loss_delta_m_penalty": 0,
        "loss_delta_av_penalty": 0,
        "loss_delta_p_penalty": 0,
        "loss_covariance_penalty": 50000,  # Seems excessive
        "loss_decorrelate_all": True,
        "loss_decorrelate_dust": True,
        "seperate_latent_training": True,
        "seperate_z_latent_training": False,
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
    legacy_pae_result_factory: "Callable[[dict[str, Any], dict[str, Any]], list[PAEStepResult]]",
) -> "list[PAEStepResult]":
    return legacy_pae_result_factory(data_params, pae_params)
