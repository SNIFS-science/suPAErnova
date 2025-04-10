from typing import ClassVar

from pydantic import PositiveInt, PositiveFloat

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.pae.model import Backend


class NFlowModelConfig(StepConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "nflow_model"
    required_steps: ClassVar[list[str]] = [PAEStepConfig.id]

    # === Required ===
    backend: Backend
    debug: bool = False

    # === Optional ===
    seed: int = 12345
    batch_size: PositiveInt = 32

    save_best: bool = False

    epochs: PositiveInt = 1000
    learning_rate: PositiveFloat = 0.001
    batch_normalisation: bool = False

    n_hidden_units: PositiveInt
    n_layers: PositiveInt
    physical_latents: bool
