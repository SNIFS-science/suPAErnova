from typing import ClassVar

from pydantic import PositiveInt, PositiveFloat

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
    random_initial_positions: bool = True
    tolerance: PositiveFloat = 0.01
    max_iterations: PositiveInt = 2500

    n_chains_early: PositiveInt = 10
    n_chains_mid: PositiveInt = 10
    n_chains_final: PositiveInt = 10

    train_delta_m: bool = True
    delta_m_mean: float = 0.0
    delta_m_std: float = 0.1

    train_delta_p: bool = True
    delta_p_mean: float = 0.0
    delta_p_std: float = 0.01
