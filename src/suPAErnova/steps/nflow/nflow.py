from typing import TYPE_CHECKING, ClassVar, override

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.pae.tf import TFPAEModel
from suPAErnova.steps.pae.tch import TCHPAEModel
from suPAErnova.steps.nflow.tf import TFNFlowModel
from suPAErnova.steps.nflow.tch import TCHNFlowModel
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
        self.nflow_models: list[NFlowModel[Model]]

    @override
    def _setup(self, *, pae: "PAEStep") -> None:
        # --- Previous Step Variables ---
        self.pae = pae

        # --- Models ---
        self.nflow_models = [
            NFlowModel[ModelMap[model.__class__]](model)
            for model in self.options.models
        ]

        for i, nflow_model in enumerate(self.nflow_models):
            compat_pae_models = [
                pae_model
                for pae_model in self.pae.pae_models
                if pae_model.__class__
                == CompatabilityMap[self.nflow_models[i].__class__]
            ]
            print(compat_pae_models)
            nflow_model.setup(data=self.pae.data, pae=self.pae.pae_models[i])


NFlowStep.register_step()
