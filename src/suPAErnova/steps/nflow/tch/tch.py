# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.nflow.model import NFlowModel
    from suPAErnova.configs.steps.nflow.tch import TCHNFlowModelConfig


class TCHNFlowModel:
    def __init__(
        self,
        config: "NFlowModel[TCHNFlowModel]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        self.name: str = f"{config.name.split()[-1]}NFlowModel"
        super().__init__(*args, **kwargs)

        # --- Config ---
        options = cast("TCHNFlowModelConfig", config.options)
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force

        # --- Latent Dimensions ---
        self.n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents
        self.n_latents: int = self.n_physical + self.n_zs

        # --- Layers ---
        # self.encoder: TFPAEEncoder = TFPAEEncoder(options, config.name)
        # self.decoder: TFPAEDecoder = TFPAEDecoder(options, config.name)
        # self.decoder.wl_dim = config.wl_dim

        # --- Training ---
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}"

        # Data Offsets
        self.phase_offset_scale: Literal[0, -1] | float = options.phase_offset_scale
        self.amplitude_offset_scale: float = options.amplitude_offset_scale
        self.mask_fraction: float = options.mask_fraction

        # self._scheduler: type[ks.optimizers.schedules.LearningRateSchedule] = (
        #     options.scheduler_cls
        # )
        # self._optimiser: type[ks.optimizers.Optimizer] = options.optimiser_cls
        # self._loss: ks.losses.Loss = options.loss_cls()

        self.stage: Stage
        # self.latents_z_mask: ITensor[S["n_latents"]]
        # self.latents_physical_mask: ITensor[S["n_latents"]]
        self._epoch: int = 0

        # --- Loss ---
        # self._loss_terms: FTensor[S["4"]]
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

        # --- Metrics ---
        # self.loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss")
        # self.recon_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="recon_loss")
        # self.model_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="model_loss")
        # self.cov_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="cov_loss")

    def save_checkpoint(self, savepath: "Path") -> None:
        pass

    def load_checkpoint(
        self, loadpath: "Path", *, reset_weights: bool | None = None
    ) -> None:
        pass

    def train_model(
        self,
        stage: "Stage",
    ) -> None:
        self.stage = stage
