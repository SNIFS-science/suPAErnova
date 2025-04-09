from typing import TYPE_CHECKING, ClassVar, get_args, override

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.pae.tf import TFPAEModel
from suPAErnova.steps.pae.tch import TCHPAEModel
from suPAErnova.steps.nflow.tf import TFNFlowModel
from suPAErnova.steps.nflow.tch import TCHNFlowModel
from suPAErnova.steps.pae.model import PAEModel
from suPAErnova.steps.nflow.model import NFlowModel
from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.nflow.tf import TFNFlowModelConfig
from suPAErnova.configs.steps.nflow.tch import TCHNFlowModelConfig

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.pae import PAEStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig

ModelConfig = TFNFlowModelConfig | TCHNFlowModelConfig
Model = TFNFlowModel | TCHNFlowModel
PAE = TFPAEModel | TCHPAEModel
ModelMap: "dict[type[ModelConfig], type[Model]]" = {
    TFNFlowModelConfig: TFNFlowModel,
    TCHNFlowModelConfig: TCHNFlowModel,
}
CompatabilityMap: "dict[type[ModelConfig], type[PAE]]" = {
    TFNFlowModelConfig: TFPAEModel,
    TCHNFlowModelConfig: TCHPAEModel,
}


class NFlowStep(SNPAEStep[NFlowStepConfig]):
    # Class Variables
    id: ClassVar["str"] = "nflow"

    def __init__(self, config: "NFlowStepConfig") -> None:
        # --- Superclass Variables ---
        self.options: NFlowStepConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Previous Step Variables ---
        self.pae: PAEStep

        # --- Setup Variables ---
        self.nflow_models: list[NFlowModel[Model]] = []

    @override
    def _setup(self, *, pae: "PAEStep") -> None:
        # --- Previous Step Variables ---
        self.pae = pae

        # --- Models ---
        for model in self.options.models:
            compat_pae_models = [
                pae_model
                for pae_model in self.pae.pae_models
                if get_args(pae_model.__orig_class__)[0]
                == CompatabilityMap[model.__class__]
            ]
            if len(compat_pae_models) == 0:
                err = f"{self.name} has no compatable PAE models with backend: {CompatabilityMap[model.__class__]}"
                raise ValueError(err)

            for pae_model in compat_pae_models:
                nflow_model = NFlowModel[ModelMap[model.__class__]](model)
                nflow_model.setup(pae=pae_model)
                self.nflow_models.append(nflow_model)

    @override
    def _completed(self) -> bool:
        return all(nflow_model.completed() for nflow_model in self.nflow_models)

    @override
    def _load(self) -> None:
        for nflow_model in self.nflow_models:
            nflow_model.load()

    @override
    def _run(self) -> None:
        for nflow_model in self.nflow_models:
            nflow_model.run()

    @override
    def _result(self) -> None:
        for nflow_model in self.nflow_models:
            nflow_model.result()

    @override
    def _analyse(self) -> None:
        for nflow_model in self.nflow_models:
            nflow_model.analyse()


NFlowStep.register_step()
