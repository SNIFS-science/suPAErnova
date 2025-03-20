# Copyright 2025 Patrick Armstrong

from typing import ClassVar

from suPAErnova.configs.steps import StepConfig


class NFlowStepConfig(StepConfig):
    # Class Vars
    name: ClassVar["str"] = "nflow"
    required_steps: ClassVar["list[str]"] = ["pae"]

    # Required

    # Optional


NFlowStepConfig.register_step()
