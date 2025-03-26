# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, Any, override

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.pae.model import PAEModelConfig

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.config import GlobalConfig
    from suPAErnova.steps.pae.tf.model import TFPAEModel
    from suPAErnova.steps.pae.tch.model import TCHPAEModel

    Model = TFPAEModel | TCHPAEModel


class PAEModel(SNPAEStep[PAEModelConfig]):
    def __init__(self, model_cls: "type[Model]", config: "PAEModelConfig") -> None:
        self.model_cls: type[Model] = model_cls

        # --- Superclass Variables ---
        self.options: PAEModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Previous Step Variables ---
        self.data: DataStep
        self.nspec_dim: int
        self.wl_dim: int
        self.phase_dim: int

        # --- Config Variables ---
        # Required

        # Optional
        self.colourlaw: Path | None

        # --- Setup Variables ---
        self.model_config: dict[str, Any]

        # --- Run Variables ---

    @override
    def _setup(self, *, data: "DataStep") -> None:
        # --- Previous Step Variables ---
        self.data = data
        self.nspec_dim = self.data.nspec_dim
        self.wl_dim = self.data.wl_dim
        self.phase_dim = 1

        # --- Config Variables ---
        # Required

        # Optional
        self.colourlaw = self.options.colourlaw

        # --- Computed Variables ---
        self.model_config = {
            "nspec_dim": self.nspec_dim,
            "wl_dim": self.wl_dim,
            "phase_dim": self.phase_dim,
            **self.options.model_dump(),
        }

    #
    # === PAEModel Specific Functions ===
    #

    def model(self) -> "Model":
        return self.model_cls(self.model_config)
