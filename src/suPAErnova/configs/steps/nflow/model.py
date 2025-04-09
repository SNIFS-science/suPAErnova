from typing import ClassVar

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.pae.pae import PAEStepConfig
from suPAErnova.configs.steps.pae.model import Backend


class NFlowModelConfig(StepConfig):
    # --- Class Variables ---
    id: ClassVar["str"] = "pae_model"
    required_steps: ClassVar["list[str]"] = [PAEStepConfig.id]

    # === Required ===
    backend: "Backend"
    debug: bool = False
