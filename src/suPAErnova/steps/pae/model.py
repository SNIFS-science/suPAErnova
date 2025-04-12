# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, final, get_args, override
from pathlib import Path  # noqa: TC003

import numpy as np
from pydantic import (  # noqa: TC002
    BaseModel,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
)

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.data import SNPAEData  # noqa: TC001

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig

    from .pae import Model, ModelConfig


class Stage(BaseModel):
    stage: PositiveInt
    name: str
    fname: str
    savepath: Path | None = None
    loadpath: Path | None = None

    epochs: PositiveInt
    debug: bool

    learning_rate: PositiveFloat
    learning_rate_decay_steps: PositiveInt
    learning_rate_decay_rate: PositiveFloat
    learning_rate_weight_decay_rate: PositiveFloat

    train_data: SNPAEData
    test_data: SNPAEData
    val_data: SNPAEData

    moving_means: list[float]


class PAEModel[M: "Model", C: "ModelConfig"](SNPAEStep[C]):
    id: ClassVar[str] = "pae_model"

    def __init__(self, config: C) -> None:
        self.model: M

        # --- Superclass Variables ---
        self.options: C
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variables ---
        # Required
        self.physical_latents: bool
        self.n_physical: int
        self.n_zs: int
        self.n_pae_latents: int

        self.seperate_latent_training: bool
        self.seperate_z_latent_training: bool

        self.debug: bool

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
        self.stage_delta_av: Stage
        self.stage_zs: list[Stage]
        self.stage_delta_m: Stage
        self.stage_delta_p: Stage
        self.stage_final: Stage
        self.run_stages: list[Stage]

    def prep_model(self, *, force: bool = False) -> M:
        if not force:
            try:
                return self.model
            except AttributeError:
                pass
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)
        return self.model

    @override
    def _setup(
        self,
        *,
        data: "DataStep",
        train_data: SNPAEData,
        test_data: SNPAEData,
        val_data: SNPAEData,
    ) -> None:
        # --- Config Variables ---
        # Required
        self.physical_latents = self.options.physical_latents
        self.n_physical = 3 if self.options.physical_latents else 0
        self.n_zs = self.options.n_z_latents
        self.n_pae_latents = self.n_physical + self.n_zs

        self.seperate_latent_training = self.options.seperate_latent_training
        self.seperate_z_latent_training = self.options.seperate_z_latent_training

        self.debug = self.options.debug

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
        stage_data = {
            "train_data": self.train_data,
            "test_data": self.test_data,
            "val_data": self.val_data,
            "moving_means": [0 for _ in range(self.n_pae_latents)],
            "debug": self.debug,
        }

        self.stage_delta_av = Stage.model_validate({
            "stage": 1,
            "name": "ΔAᵥ",
            "fname": "delta_av",
            "epochs": self.options.delta_av_epochs,
            "learning_rate": self.options.delta_av_lr,
            "learning_rate_decay_steps": self.options.delta_av_lr_decay_steps,
            "learning_rate_decay_rate": self.options.delta_av_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.delta_av_lr_weight_decay_rate,
            **stage_data,
        })

        z0 = 2 if self.physical_latents else 1
        self.stage_zs = [
            Stage.model_validate({
                "stage": z0 + i,
                "name": f"z{i + 1}",
                "fname": f"z{i + 1}",
                "epochs": self.options.zs_epochs,
                "learning_rate": self.options.zs_lr,
                "learning_rate_decay_steps": self.options.zs_lr_decay_steps,
                "learning_rate_decay_rate": self.options.zs_lr_decay_rate,
                "learning_rate_weight_decay_rate": self.options.zs_lr_weight_decay_rate,
                **stage_data,
            })
            for i in range(self.n_zs)
        ]

        self.stage_delta_m = Stage.model_validate({
            "stage": z0 + self.n_zs,
            "name": "Δℳ",
            "fname": "delta_m",
            "epochs": self.options.delta_m_epochs,
            "learning_rate": self.options.delta_m_lr,
            "learning_rate_decay_steps": self.options.delta_m_lr_decay_steps,
            "learning_rate_decay_rate": self.options.delta_m_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.delta_m_lr_weight_decay_rate,
            **stage_data,
        })

        self.stage_delta_p = Stage.model_validate({
            "stage": z0 + self.n_zs + 1,
            "name": "Δ𝓅",
            "fname": "delta_p",
            "epochs": self.options.delta_p_epochs,
            "learning_rate": self.options.delta_p_lr,
            "learning_rate_decay_steps": self.options.delta_p_lr_decay_steps,
            "learning_rate_decay_rate": self.options.delta_p_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.delta_p_lr_weight_decay_rate,
            **stage_data,
        })

        self.stage_final = Stage.model_validate({
            "stage": self.n_pae_latents,
            "name": "Final",
            "fname": "final",
            "epochs": self.options.final_epochs,
            "learning_rate": self.options.final_lr,
            "learning_rate_decay_steps": self.options.final_lr_decay_steps,
            "learning_rate_decay_rate": self.options.final_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.final_lr_weight_decay_rate,
            **stage_data,
        })

        if self.physical_latents:
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
        self.prep_model()
        final_stage = self.run_stages[-1]
        final_savepath = (
            self.paths.out / self.model.name / final_stage.fname / self.model.model_path
        )

        if not final_savepath.exists():
            self.log.debug(
                f"{self.name} has not completed as {final_savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self.prep_model()

        final_stage = self.run_stages[-1]
        self.model.stage = final_stage
        final_savepath = self.paths.out / self.model.name / final_stage.fname

        self.log.debug(f"Loading final PAE model weights from {final_savepath}")
        self.model.load_checkpoint(final_savepath, reset_weights=False)

    @override
    def _run(self) -> None:
        savepath: Path | None = None
        for i, stage in enumerate(self.run_stages):
            self.prep_model(force=True)
            self.log.debug(f"Starting Stage {i}: {stage.name}")
            if savepath is not None:
                stage.loadpath = savepath
            savepath = self.paths.out / self.model.name / stage.fname
            stage.savepath = savepath

            model_path = savepath / self.model.model_path
            weights_path = savepath / self.model.weights_path
            if model_path.exists() and not self.force:
                # Don't retrain stages if you don't need to
                self.log.debug(f"Loading stage weights from {weights_path}")
                self.model.stage = stage
                self.model.load_checkpoint(savepath, reset_weights=False)
            else:
                self.model.train_model(stage)
            self.model.save_checkpoint(savepath)

    @override
    def _result(self) -> None:
        self.prep_model()
        final_stage = self.run_stages[-1]
        self.model.stage = final_stage
        final_savepath = self.paths.out / self.model.name / final_stage.fname

        self.log.debug(f"Saving final PAE model weights to {final_savepath}")
        self.model.save_checkpoint(final_savepath)

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

            mask = (redshift_mask & phase_mask).astype(np.int32)
            data.mask *= mask
            setattr(self, f"{mask_type}_data", data)
