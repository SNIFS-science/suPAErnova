# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, final, override

from pydantic import (  # noqa: TC002
    BaseModel,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
)

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.pae.tf import TFPAEModel
from suPAErnova.steps.pae.tch import TCHPAEModel
from suPAErnova.configs.steps.pae import ModelConfig
from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig

if TYPE_CHECKING:
    from logging import Logger

    import numpy as np
    from numpy import typing as npt

    from suPAErnova.steps.pae import Model
    from suPAErnova.steps.data import DataStep, SNPAEData
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig


class Stage(BaseModel):
    name: str
    stage: "PositiveInt"

    training: bool
    epochs: "PositiveInt"
    learning_rate: "PositiveFloat"
    moving_means: list[float]


@final
class PAEModel(SNPAEStep[ModelConfig]):
    id: ClassVar["str"] = "pae_model"

    def __init__(self, config: "ModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: ModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Backend Variables ---
        self.model_cls: type[Model] = (
            TFPAEModel if isinstance(self.options, TFPAEModelConfig) else TCHPAEModel
        )

        # --- Config Variables ---
        # Required
        self.n_physical: int
        self.n_zs: int
        self.n_latents: int

        self.seperate_latent_training: bool
        self.seperate_z_latent_training: bool

        # Optional
        self.min_train_redshift: NonNegativeFloat
        self.max_train_redshift: NonNegativeFloat
        self.min_test_redshift: NonNegativeFloat
        self.max_test_redshift: NonNegativeFloat
        self.min_val_redshift: NonNegativeFloat
        self.max_val_redshift: NonNegativeFloat

        self.min_train_phase: NonNegativeFloat
        self.max_train_phase: NonNegativeFloat
        self.min_test_phase: NonNegativeFloat
        self.max_test_phase: NonNegativeFloat
        self.min_val_phase: NonNegativeFloat
        self.max_val_phase: NonNegativeFloat

        # --- Previous Step Variables ---
        self.data: DataStep
        self.train_data: SNPAEData
        self.test_data: SNPAEData
        self.val_data: SNPAEData

        self.nspec_dim: int
        self.wl_dim: int
        self.phase_dim: int

        # --- Setup Variables ---
        self.mask_train: npt.NDArray[np.bool_]
        self.mask_test: npt.NDArray[np.bool_]
        self.mask_val: npt.NDArray[np.bool_]

        self.stage_delta_av: Stage
        self.stage_zs: list[Stage]
        self.stage_delta_m: Stage
        self.stage_delta_p: Stage
        self.stage_final: Stage
        self.run_stages: list[Stage]

    @override
    def _setup(
        self,
        *,
        data: "DataStep",
        train_data: "SNPAEData",
        test_data: "SNPAEData",
        val_data: "SNPAEData",
    ) -> None:
        # --- Config Variables ---
        # Required
        self.n_physical = 3 if self.options.physical_latents else 0
        self.n_zs = self.options.n_z_latents
        self.n_latents = self.n_physical + self.n_zs

        self.seperate_latent_training = self.options.seperate_latent_training
        self.seperate_z_latent_training = self.options.seperate_z_latent_training

        # Optional
        self.min_train_redshift = self.options.min_train_redshift
        self.max_train_redshift = self.options.max_train_redshift
        self.min_test_redshift = self.options.min_test_redshift
        self.max_test_redshift = self.options.max_test_redshift
        self.min_val_redshift = self.options.min_val_redshift
        self.max_val_redshift = self.options.max_val_redshift

        self.min_train_phase = self.options.min_train_phase
        self.max_train_phase = self.options.max_train_phase
        self.min_test_phase = self.options.min_test_phase
        self.max_test_phase = self.options.max_test_phase
        self.min_val_phase = self.options.min_val_phase
        self.max_val_phase = self.options.max_val_phase

        # --- Previous Step Variables ---
        self.data = data
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.setup_data_masks()

        self.nspec_dim = self.data.nspec_dim
        self.wl_dim = self.data.wl_dim
        self.phase_dim = 1

        # --- Stages ---
        self.stage_delta_av = Stage.model_validate({
            "name": "ΔAᵥ",
            "stage": 1,
            "training": True,
            "epochs": self.options.delta_av_epochs,
            "learning_rate": self.options.delta_av_lr,
            "moving_means": [],
        })

        z0 = 2 if self.n_physical > 0 else 1
        self.stage_zs = [
            Stage.model_validate({
                "name": f"z{i}",
                "stage": z0 + i,
                "training": True,
                "epochs": self.options.zs_epochs,
                "learning_rate": self.options.zs_lr,
                "moving_means": [],
            })
            for i in range(self.n_zs)
        ]

        self.stage_delta_m = Stage.model_validate({
            "name": "Δℳ",
            "stage": z0 + self.n_zs,
            "training": True,
            "epochs": self.options.delta_m_epochs,
            "learning_rate": self.options.delta_m_lr,
            "moving_means": [],
        })

        self.stage_delta_m = Stage.model_validate({
            "name": "Δℳ",
            "stage": z0 + self.n_zs,
            "training": True,
            "epochs": self.options.delta_m_epochs,
            "learning_rate": self.options.delta_m_lr,
            "moving_means": [],
        })

        self.stage_delta_p = Stage.model_validate({
            "name": "Δ𝓅",
            "stage": z0 + self.n_zs + 1,
            "training": True,
            "epochs": self.options.delta_p_epochs,
            "learning_rate": self.options.delta_p_lr,
            "moving_means": [],
        })

        self.stage_final = Stage.model_validate({
            "name": "Final",
            "stage": self.n_latents,
            "training": True,
            "epochs": self.options.final_epochs,
            "learning_rate": self.options.final_lr,
            "moving_means": [],
        })

        if self.n_physical > 0:
            self.run_stages = [
                self.stage_delta_av,
                *self.stage_zs,
                self.stage_delta_m,
                self.stage_delta_p,
                self.stage_final,
            ]
        else:
            self.run_stages = [*self.stage_zs, self.stage_final]

    @override
    def _completed(self) -> bool:
        return False

    @override
    def _load(self) -> None:
        pass

    @override
    def _run(self) -> None:
        stage = 1 if self.seperate_latent_training else self.n_latents

    @override
    def _result(self) -> None:
        pass

    @override
    def _analyse(self) -> None:
        pass

    #
    # === PAEModel Specific Functions ===
    #

    def setup_data_masks(self) -> None:
        for mask_type in ["train", "test", "val"]:
            data: SNPAEData = getattr(self, f"{mask_type}_data")
            min_redshift: float = getattr(self, f"min_{mask_type}_redshift")
            max_redshift: float = getattr(self, f"max_{mask_type}_redshift")
            redshift_mask = (data.redshift >= min_redshift) & (
                data.redshift <= max_redshift
            )

            min_phase: float = getattr(self, f"min_{mask_type}_phase")
            max_phase: float = getattr(self, f"max_{mask_type}_phase")
            phase_mask = (data.phase >= min_phase) & (data.phase <= max_phase)

            mask = f"mask_{mask_type}"
            setattr(self, mask, redshift_mask & phase_mask)

    def model(self) -> "Model":
        return self.model_cls(self)
