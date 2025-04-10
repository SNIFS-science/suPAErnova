# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    cast,
)

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.pae.tch import TCHPAEModel
    from suPAErnova.configs.steps.nflow.tch import TCHNFlowModelConfig

    from .model import NFlowModel

    NFlowInputs = Any
    NFlowOutputs = Any


class TCHNFlowModel:
    def __init__(
        self,
        config: "NFlowModel[TCHNFlowModel, TCHNFlowModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name: str
        # --- Config ---
        options = cast("TCHNFlowModelConfig", config.options)
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.debug: bool = options.debug
        self.pae: TCHPAEModel = cast("TCHPAEModel", config.pae.model)

        # --- Training ---
        self.built: bool = False
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

        self.n_hidden_units: int = options.n_hidden_units
        self.n_layers: int = options.n_layers
        self.batch_normalisation: bool = options.batch_normalisation
        self.physical_latents: bool = options.physical_latents

        self.learning_rate: float = options.learning_rate
        self.epochs: int = options.epochs
        self._epoch: int = 0

        self.data: "NFlowInputs"

        # --- Latent Dimensions ---
        self.n_latents: int = self.pae.n_zs
        if self.physical_latents:
            self.n_latents += 1

    def call(
        self,
        _inputs: "NFlowInputs",
        training: bool | None = None,
        _mask: "None" = None,
    ) -> "NFlowOutputs":
        training = False if training is None else training
        return None

    def train_model(
        self,
        *,
        _savepath: "Path | None" = None,
    ) -> None:
        pass

    def build_model(
        self,
    ) -> None:
        if not self.built:
            self.built = True

    def save_checkpoint(self, _savepath: "Path") -> None:
        pass

    def load_checkpoint(self, _loadpath: "Path") -> None:
        pass
