# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, ClassVar, override

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.pae.model import PAEModel
from suPAErnova.configs.steps.pae import PAEStepConfig

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.config import GlobalConfig
    from suPAErnova.configs.steps.pae import Backend
    from suPAErnova.steps.pae.tf.model import TFPAEModel
    from suPAErnova.steps.pae.tch.model import TCHPAEModel

    Model = TFPAEModel | TCHPAEModel


class PAEStep(SNPAEStep[PAEStepConfig]):
    # Class Variables
    name: ClassVar["str"] = "pae"

    def __init__(self, config: "PAEStepConfig") -> None:
        # --- Superclass Variables ---
        self.options: PAEStepConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Previous Step Variables ---

        # --- Config Variables ---
        # Required
        self.backend: Backend

        # Optional

        # --- Setup Variables ---
        self.pae_model: PAEModel

    @override
    def _setup(self, *, data: "DataStep") -> None:
        # --- Previous Step Variables ---

        # --- Config Variables ---
        # Required
        self.backend = self.options.backend

        # Optional

        # --- Computed Variables ---
        self.pae_model = PAEModel(self.options.model)
        self.pae_model.setup(data=data)

    @override
    def _completed(self) -> bool:
        return False

    @override
    def _load(self) -> None:
        pass

    @override
    def _run(self) -> None:
        pass

    @override
    def _result(self) -> None:
        pass

    @override
    def _analyse(self) -> None:
        pass


PAEStep.register_step()
