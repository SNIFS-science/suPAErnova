# Copyright 2025 Patrick Armstrong

from typing import ClassVar

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.pae import PAEStepConfig


class NFlowStepConfig(StepConfig):
    # Class Vars
    id: ClassVar["str"] = "nflow"
    required_steps: ClassVar["list[str]"] = [PAEStepConfig.id]

    # Required

    # Optional


NFlowStepConfig.register_step()
