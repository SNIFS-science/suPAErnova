# Copyright 2025 Patrick Armstrong

from typing import Self

from pydantic import computed_field, model_validator

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs import SNPAEConfig
from suPAErnova.steps.pae import PAEStep  # noqa: TC001
from suPAErnova.steps.data import DataStep  # noqa: TC001
from suPAErnova.steps.nflow import NFlowStep  # noqa: TC001
from suPAErnova.configs.steps import StepConfig  # noqa: TC001
from suPAErnova.steps.posterior import PosteriorStep  # noqa: TC001
from suPAErnova.configs.steps.pae import PAEStepConfig  # noqa: TC001
from suPAErnova.configs.steps.data import DataStepConfig  # noqa: TC001
from suPAErnova.configs.steps.nflow import NFlowStepConfig  # noqa: TC001
from suPAErnova.configs.steps.posterior import PosteriorStepConfig  # noqa: TC001


class InputConfig(SNPAEConfig):
    """User-defined input configuration which controls the behaviour of SuPAErnova."""

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
    def step_configs(self) -> list["StepConfig"]:
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
    def steps(self) -> list["SNPAEStep[StepConfig]"]:
        return [SNPAEStep.steps[step.name](step) for step in self.step_configs]

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        for step_config in self.step_configs:
            for required_step in step_config.required_steps:
                if getattr(self, required_step) is None:
                    err = (
                        f"{step_config.name} requires that {required_step} is run first"
                    )
                    raise ValueError(err)
        return self

    def require(self, step_name: str) -> SNPAEStep[StepConfig]:
        step = getattr(self, step_name + "_step")
        if step is None:
            err = f"{step_name} has not yet run"
            raise ValueError(err)
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
            setattr(self, step.name + "_step", step)
