# Copyright 2025 Patrick Armstrong

from typing import ClassVar

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.nflow import NFlowStepConfig


class PosteriorStepConfig(StepConfig):
    # Class Vars
    name: ClassVar["str"] = "posterior"
    required_steps: ClassVar["list[str]"] = [NFlowStepConfig.name]

    # Required

    # Optional


PosteriorStepConfig.register_step()
