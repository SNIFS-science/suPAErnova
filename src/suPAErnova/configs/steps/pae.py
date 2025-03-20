# Copyright 2025 Patrick Armstrong

from typing import ClassVar

from suPAErnova.configs.steps import StepConfig


class PAEStepConfig(StepConfig):
    # Class Vars
    name: ClassVar["str"] = "pae"
    required_steps: ClassVar["list[str]"] = ["data"]

    # Required

    # Optional


PAEStepConfig.register_step()
