# Copyright 2025 Patrick Armstrong
"""User-defined input configuration which controls the behaviour of SuPAErnova."""

from suPAErnova.configs import SNPAEConfig
from suPAErnova.configs.steps import StepConfig  # NOQA: TC001


class InputConfig(SNPAEConfig):
    """User-defined input configuration which controls the behaviour of SuPAErnova."""

    steps: list["StepConfig"]
