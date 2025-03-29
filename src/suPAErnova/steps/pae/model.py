# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, final, override

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.pae import ModelConfig
from suPAErnova.steps.pae.tf.model import TFPAEModel
from suPAErnova.steps.pae.tch.model import TCHPAEModel
from suPAErnova.configs.steps.pae.tf.model import TFPAEModelConfig

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.pae import Model
    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.config import GlobalConfig


@final
class PAEModel(SNPAEStep[ModelConfig]):
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

        # --- Previous Step Variables ---
        self.data: DataStep
        self.nspec_dim: int
        self.wl_dim: int
        self.phase_dim: int

    @override
    def _setup(self, *, data: "DataStep") -> None:
        # --- Previous Step Variables ---
        self.data = data
        self.nspec_dim = self.data.nspec_dim
        self.wl_dim = self.data.wl_dim
        self.phase_dim = 1

    #
    # === PAEModel Specific Functions ===
    #

    def model(self) -> "Model":
        return self.model_cls(self)
