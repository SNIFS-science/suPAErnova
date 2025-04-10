# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

if TYPE_CHECKING:
    from typing import (
        Any,
        Literal,
    )
    from logging import Logger
    from pathlib import Path

    from numpy import typing as npt

    from suPAErnova.configs.steps.pae.tch import TCHPAEModelConfig

    from .model import Stage, PAEModel


class TCHPAEEncoder:
    def __init__(
        self,
        options: "TCHPAEModelConfig",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, **kwargs)

        # --- Config Params ---
        n_physical: "Literal[0, 3]" = 3 if options.physical_latents else 0
        n_zs: int = options.n_z_latents
        self.n_latents: int = n_physical + n_zs

        self.stage_num: int

        self.encode_dims: list[int] = options.encode_dims

        self.dropout: float = options.dropout
        self.batch_normalisation: bool = options.batch_normalisation


class TCHPAEDecoder:
    def __init__(
        self, options: "TCHPAEModelConfig", name: str, *args: "Any", **kwargs: "Any"
    ) -> None:
        super().__init__(*args, **kwargs)
        # --- Config Params ---
        self.n_physical: "Literal[0, 3]" = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents

        self.wl_dim: int
        self.decode_dims: list[int] = options.decode_dims

        self.batch_normalisation: bool = options.batch_normalisation

        colourlaw = options.colourlaw
        if colourlaw is not None:
            _, colourlaw = np.loadtxt(colourlaw, unpack=True)
        self.colourlaw: "npt.NDArray[np.float64] | None" = colourlaw


class TCHPAEModel:
    def __init__(
        self,
        config: "PAEModel[TCHPAEModel, TCHPAEModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, **kwargs)
        # --- Config ---
        options = cast("TCHPAEModelConfig", config.options)
        self.log: "Logger" = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force

        # --- Latent Dimensions ---
        self.n_physical: "Literal[0, 3]" = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents
        self.n_latents: int = self.n_physical + self.n_zs

        # --- Layers ---
        self.encoder: TCHPAEEncoder = TCHPAEEncoder(options, config.name)
        self.decoder: TCHPAEDecoder = TCHPAEDecoder(options, config.name)
        self.decoder.wl_dim = config.wl_dim

        # --- Training ---
        self.built: bool = False
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

        # Data Offsets
        self.phase_offset_scale: "Literal[0, -1] | float" = options.phase_offset_scale
        self.amplitude_offset_scale: float = options.amplitude_offset_scale
        self.mask_fraction: float = options.mask_fraction

        self.stage: Stage
        self._epoch: int = 0

        # --- Loss ---
        self.loss_residual_penalty: float = options.loss_residual_penalty

        self.loss_delta_av_penalty: float = options.loss_delta_av_penalty
        self.loss_delta_m_penalty: float = options.loss_delta_m_penalty
        self.loss_delta_p_penalty: float = options.loss_delta_p_penalty
        self.loss_physical_penalty: float = sum((
            self.loss_delta_av_penalty,
            self.loss_delta_m_penalty,
            self.loss_delta_p_penalty,
        ))  # Only calculate physical latent penalties if at least one penalty scaling is > 0

        self.loss_covariance_penalty: float = options.loss_covariance_penalty
        self.loss_decorrelate_all: bool = options.loss_decorrelate_all
        self.loss_decorrelate_dust: bool = options.loss_decorrelate_dust

    def train_model(
        self,
        _stage: "Stage",
    ) -> None:
        pass

    def build_model(self) -> None:
        if not self.built:
            self.built = True

    def save_checkpoint(self, _savepath: "Path") -> None:
        pass

    def load_checkpoint(
        self, _loadpath: "Path", *, _reset_weights: bool | None = None
    ) -> None:
        pass
