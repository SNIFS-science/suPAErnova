# Copyright 2025 Patrick Armstrong

from typing import Self, Literal, ClassVar, Annotated
from pathlib import Path
import itertools

from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
    field_validator,
    model_validator,
)

from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.backends import AbstractModelConfig


class PAEModelConfig(AbstractModelConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "pae_model"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id]

    # === Required ===
    debug: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        for var in ["redshift", "phase"]:
            for data in ["train", "test", "val"]:
                min_bound = getattr(self, f"min_{data}_{var}")
                max_bound = getattr(self, f"max_{data}_{var}")
                if min_bound >= max_bound:
                    err = f"`max_{data}_{var}`: {max_bound} is not strictly greater than `min_{data}_{var}`: {min_bound}"
                    self._raise(err)
        return self

    # --- Network Design ---
    architecture: Literal["dense", "convolutional"]
    encode_dims: list[PositiveInt]
    decode_dims: list[PositiveInt] = []

    @field_validator("encode_dims", mode="before")
    @classmethod
    def validate_encode_dims(cls, value: list[int]) -> list[int]:
        if len(value) == 0:
            err = "`encode_dims` can not be empty"
            cls._raise(err)
        if not all(x > y for x, y in itertools.pairwise(value)):
            err = f"`encode_dims`: {value} is not monotonically decreasing"
            cls._raise(err)
        return value

    @model_validator(mode="after")
    def validate_decode_dims(self) -> Self:
        if len(self.decode_dims) == 0:
            self.decode_dims = list(reversed(self.encode_dims))
        if not all(x < y for x, y in itertools.pairwise(self.decode_dims)):
            err = f"`decode_dims`: {self.decode_dims} is not monotonically decreasing"
            self._raise(err)
        return self

    physical_latents: bool
    n_z_latents: PositiveInt

    @model_validator(mode="after")
    def validate_n_latents(self) -> Self:
        if not self.physical_latents and self.n_z_latents == 0:
            err = "You must specify either non-zero `n_z_latents`, or `physical_latents=True`. With both `physical_latents=False` and `n_z_latents=0, there will be no latents to train at all!"
            self._raise(err)
        return self

    # --- Training ---
    # Overfitting
    batch_normalisation: bool
    dropout: Annotated[float, Field(ge=0, le=1)]

    # Latent training
    seperate_latent_training: bool
    seperate_z_latent_training: bool

    # === Optional ===
    seed: int = 12345
    batch_size: PositiveInt = 32

    save_best: bool = False

    # --- Data ---
    min_train_redshift: float = -float("inf")
    max_train_redshift: float = +float("inf")
    min_test_redshift: float = -float("inf")
    max_test_redshift: float = +float("inf")
    min_val_redshift: float = -float("inf")
    max_val_redshift: float = +float("inf")

    min_train_phase: float = -float("inf")
    max_train_phase: float = +float("inf")
    min_test_phase: float = -float("inf")
    max_test_phase: float = +float("inf")
    min_val_phase: float = -float("inf")
    max_val_phase: float = +float("inf")

    # --- Data Offsets ---
    phase_offset_scale: float = 0
    amplitude_offset_scale: NonNegativeFloat = 0
    mask_fraction: Annotated[float, Field(ge=0, le=1)] = 0

    # --- Loss ---
    loss_residual_penalty: NonNegativeFloat = 0

    loss_delta_av_penalty: NonNegativeFloat = 0
    loss_delta_m_penalty: NonNegativeFloat = 0
    loss_delta_p_penalty: NonNegativeFloat = 0

    loss_covariance_penalty: NonNegativeFloat = 0
    loss_decorrelate_all: bool = False
    loss_decorrelate_dust: bool = False

    # --- Stages ---
    # ΔAᵥ
    delta_av_epochs: PositiveInt = 1000
    delta_av_lr: PositiveFloat = 0.005
    delta_av_lr_decay_steps: PositiveInt = 300
    delta_av_lr_decay_rate: PositiveFloat = 0.95
    delta_av_lr_weight_decay_rate: PositiveFloat = 0.00001

    # Zs
    zs_epochs: PositiveInt = 1000
    zs_lr: PositiveFloat = 0.005
    zs_lr_decay_steps: PositiveInt = 300
    zs_lr_decay_rate: PositiveFloat = 0.95
    zs_lr_weight_decay_rate: PositiveFloat = 0.00001

    # Δℳ
    delta_m_epochs: PositiveInt = 1000
    delta_m_lr: PositiveFloat = 0.005
    delta_m_lr_decay_steps: PositiveInt = 300
    delta_m_lr_decay_rate: PositiveFloat = 0.95
    delta_m_lr_weight_decay_rate: PositiveFloat = 0.00001

    # Δ𝓅
    delta_p_epochs: PositiveInt = 1000
    delta_p_lr: PositiveFloat = 0.005
    delta_p_lr_decay_steps: PositiveInt = 300
    delta_p_lr_decay_rate: PositiveFloat = 0.95
    delta_p_lr_weight_decay_rate: PositiveFloat = 0.00001

    # Final
    final_epochs: PositiveFloat = 5000
    final_lr: PositiveFloat = 0.001
    final_lr_decay_steps: PositiveInt = 300
    final_lr_decay_rate: PositiveFloat = 0.95
    final_lr_weight_decay_rate: PositiveFloat = 0.00001

    colourlaw: Path | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.colourlaw is not None:
            self.colourlaw = self.paths.resolve_path(
                self.colourlaw, relative_path=self.paths.base
            )
            if not self.colourlaw.exists():
                err = f"`colourlaw` resolved to {self.colourlaw}, which does not exist."
                self._raise(err)
        return self
