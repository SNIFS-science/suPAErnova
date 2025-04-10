# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    cast,
)

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.nflow.tch import TCHNFlowModel
    from suPAErnova.steps.posterior.model import PosteriorModel
    from suPAErnova.configs.steps.posterior.tch import TCHPosteriorModelConfig

    # === Custom Types ===
    PosteriorInputs = Any
    PosteriorOutputs = Any


class TCHPosteriorModel:
    def __init__(
        self,
        config: "PosteriorModel[TCHPosteriorModel, TCHPosteriorModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name: str
        # --- Config ---
        options = cast("TCHPosteriorModelConfig", config.options)
        self.log: "Logger" = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.debug: bool = options.debug
        self.nflow: "TCHNFlowModel" = cast("TCHNFlowModel", config.nflow.model)

        # --- Training ---
        self.built: bool = False
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

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
