# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, get_args, override

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.nflow.tf import TFNFlowModel
from suPAErnova.steps.nflow.tch import TCHNFlowModel
from suPAErnova.configs.steps.posterior import PosteriorStepConfig
from suPAErnova.configs.steps.posterior.tf import TFPosteriorModelConfig
from suPAErnova.configs.steps.posterior.tch import TCHPosteriorModelConfig

from .tf import TFPosteriorModel
from .tch import TCHPosteriorModel
from .model import PosteriorModel

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.nflow import NFlowStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig

ModelConfig = TFPosteriorModelConfig | TCHPosteriorModelConfig
Model = TFPosteriorModel | TCHPosteriorModel
NFLOW = TFNFlowModel | TCHNFlowModel
ModelMap: dict[type[ModelConfig], type[Model]] = {
    TFPosteriorModelConfig: TFPosteriorModel,
    TCHPosteriorModelConfig: TCHPosteriorModel,
}
CompatabilityMap: dict[type[ModelConfig], type[NFLOW]] = {
    TFPosteriorModelConfig: TFNFlowModel,
    TCHPosteriorModelConfig: TCHNFlowModel,
}


class PosteriorStep(SNPAEStep[PosteriorStepConfig]):
    # Class Variables
    id: ClassVar[str] = "posterior"

    def __init__(self, config: PosteriorStepConfig) -> None:
        # --- Superclass Variables ---
        self.options: PosteriorStepConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Previous Step Variables ---
        self.nflow: NFlowStep

        # --- Setup Variables ---
        self.posterior_models: list[PosteriorModel[Model, ModelConfig]] = []

    @override
    def _setup(self, *, nflow: "NFlowStep") -> None:
        # --- Previous Step Variables ---
        self.nflow = nflow

        # --- Models ---
        for model in self.options.models:
            compat_nflow_models = [
                nflow_model
                for nflow_model in self.nflow.nflow_models
                if get_args(nflow_model.__orig_class__)[0]
                == CompatabilityMap[model.__class__]
            ]
            if len(compat_nflow_models) == 0:
                err = f"{self.name} has no compatable NFlow models with backend: {CompatabilityMap[model.__class__]}"
                raise ValueError(err)

            for nflow_model in compat_nflow_models:
                posterior_model = PosteriorModel[
                    ModelMap[model.__class__], model.__class__
                ](model)
                posterior_model.setup(nflow=nflow_model)
                self.posterior_models.append(posterior_model)

    @override
    def _completed(self) -> bool:
        return all(
            posterior_model.completed() for posterior_model in self.posterior_models
        )

    @override
    def _load(self) -> None:
        for posterior_model in self.posterior_models:
            posterior_model.load()

    @override
    def _run(self) -> None:
        for posterior_model in self.posterior_models:
            posterior_model.run()

    @override
    def _result(self) -> None:
        for posterior_model in self.posterior_models:
            posterior_model.result()

    @override
    def _analyse(self) -> None:
        for posterior_model in self.posterior_models:
            posterior_model.analyse()


PosteriorStep.register_step()
