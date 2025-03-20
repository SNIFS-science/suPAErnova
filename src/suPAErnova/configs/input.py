# Copyright 2025 Patrick Armstrong
"""User-defined input configuration which controls the behaviour of SuPAErnova."""

from typing import Self

from pydantic import computed_field, model_validator

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs import SNPAEConfig
from suPAErnova.configs.steps import StepConfig  # noqa: TC001
from suPAErnova.configs.steps.pae import PAEStepConfig  # noqa: TC001
from suPAErnova.configs.steps.data import DataStepConfig  # noqa: TC001
from suPAErnova.configs.steps.nflow import NFlowStepConfig  # noqa: TC001
from suPAErnova.configs.steps.posterior import PosteriorStepConfig  # noqa: TC001

from suPAErnova.steps.data import DataStep
from suPAErnova.steps.pae import PAEStep
from suPAErnova.steps.nflow import NFlowStep
from suPAErnova.steps.posterior import PosteriorStep


class InputConfig(SNPAEConfig):
    """User-defined input configuration which controls the behaviour of SuPAErnova."""

    data: DataStepConfig | None = None
    pae: PAEStepConfig | None = None
    nflow: NFlowStepConfig | None = None
    posterior: "PosteriorStepConfig | None" = None

    @computed_field
    @property
    def step_configs(self) -> list["StepConfig"]:
        return [
            step
            for step in [self.data, self.pae, self.nflow, self.posterior]
            if step is not None
        ]

    @computed_field
    @property
    def steps(self) -> list["SNPAEStep[SNPAEConfig]"]:
        return [SNPAEStep.steps[step.name](step) for step in self.step_configs]

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        for step in self.step_configs:
            for required_step in step.required_steps:
                if getattr(self, required_step) is None:
                    err = f"{step.name} requires that {required_step} is run first"
                    raise ValueError(err)
        return self

    def run(self) -> None:
        for step in self.steps:
            step.setup()
            step.run()
            step.result()
            step.analyse()
