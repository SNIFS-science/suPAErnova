from typing import ClassVar

from pydantic import PositiveInt

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.pae.model import Backend


class PosteriorModelConfig(StepConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "posterior_model"
    required_steps: ClassVar[list[str]] = [NFlowStepConfig.id]

    # === Required ===
    backend: Backend
    debug: bool = False

    # === Optional ===
    seed: int = 12345
    batch_size: PositiveInt = 32
    save_best: bool = False
