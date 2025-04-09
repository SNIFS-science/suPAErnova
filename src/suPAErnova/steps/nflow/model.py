# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, final, get_args, override

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.nflow import ModelConfig

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.pae import Model as PAE
    from suPAErnova.steps.data import DataStep
    from suPAErnova.steps.nflow import Model
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.pae.model import PAEModel


@final
class NFlowModel[M: "Model"](SNPAEStep[ModelConfig]):
    id: ClassVar["str"] = "nflow_model"

    def __init__(self, config: "ModelConfig") -> None:
        self.model: M

        # --- Superclass Variables ---
        self.options: ModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool

        # --- Previous Step Variables ---
        self.data: DataStep
        self.pae: PAEModel[PAE]

    @override
    def _setup(
        self,
        *,
        data: "DataStep",
        pae: "PAEModel[PAE]",
    ) -> None:
        self.debug = self.options.debug
        self.data = data
        self.pae = pae
        print(self)

    @override
    def _completed(self) -> bool:
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)
        return False

    @override
    def _load(self) -> None:
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)

    @override
    def _run(self) -> None:
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)

    @override
    def _result(self) -> None:
        pass

    @override
    def _analyse(self) -> None:
        pass

    #
    # === NFlowModel Specific Functions ===
    #
