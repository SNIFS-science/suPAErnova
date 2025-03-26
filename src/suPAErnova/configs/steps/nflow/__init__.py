# Copyright 2025 Patrick Armstrong

from typing import ClassVar

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.pae import PAEStepConfig


class NFlowStepConfig(StepConfig):
    # Class Vars
    name: ClassVar["str"] = "nflow"
    required_steps: ClassVar["list[str]"] = [PAEStepConfig.name]

    # Required

    # Optional


NFlowStepConfig.register_step()
