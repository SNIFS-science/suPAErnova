# Copyright 2025 Patrick Armstrong

from typing import Self

from pydantic import computed_field, model_validator

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.pae import PAEStep
from suPAErnova.steps.data import DataStep
from suPAErnova.steps.nflow import NFlowStep
from suPAErnova.steps.posterior import PosteriorStep

from .steps import StepConfig
from .configs import SNPAEConfig
from .steps.pae import PAEStepConfig
from .steps.data import DataStepConfig
from .steps.nflow import NFlowStepConfig
from .steps.posterior import PosteriorStepConfig


class InputConfig(SNPAEConfig):
    data: DataStepConfig | None = None
    pae: PAEStepConfig | None = None
    nflow: NFlowStepConfig | None = None
    posterior: PosteriorStepConfig | None = None

    data_step: DataStep | None = None
    pae_step: PAEStep | None = None
    nflow_step: NFlowStep | None = None
    posterior_step: PosteriorStep | None = None

    @computed_field
    @property
    def step_configs(self) -> list[StepConfig]:
        return [
            step_config
            for step_config in [
                self.data,
                self.pae,
                self.nflow,
                self.posterior,
            ]
            if step_config is not None
        ]

    @computed_field
    @property
    def steps(self) -> list[SNPAEStep[StepConfig]]:
        return [SNPAEStep.steps[step.id](step) for step in self.step_configs]

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        for step_config in self.step_configs:
            for required_step in step_config.required_steps:
                if getattr(self, required_step) is None:
                    err = f"{step_config.id} requires that {required_step} is run first"
                    self._raise(err)
        return self

    def require(self, step_name: str) -> SNPAEStep[StepConfig]:
        step = getattr(self, step_name + "_step")
        if step is None:
            err = f"{step_name} has not yet run"
            self._raise(err)
        return step

    def run(self) -> None:
        for step in self.steps:
            args = []
            kwargs = {
                required_step: self.require(required_step)
                for required_step in step.options.required_steps
            }
            step.setup(*args, **kwargs)
            step.run()
            step.result()
            step.analyse()
            setattr(self, step.id + "_step", step)
