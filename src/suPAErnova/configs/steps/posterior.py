# Copyright 2025 Patrick Armstrong

from typing import ClassVar

from suPAErnova.configs.steps import StepConfig


class PosteriorStepConfig(StepConfig):
    # Class Vars
    name: ClassVar["str"] = "posterior"
    required_steps: ClassVar["list[str]"] = ["nflow"]

    # Required

    # Optional


PosteriorStepConfig.register_step()
