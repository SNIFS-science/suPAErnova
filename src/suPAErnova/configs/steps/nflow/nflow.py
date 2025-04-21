# Copyright 2025 Patrick Armstrong

from typing import Any, ClassVar
import importlib
from collections.abc import Callable

from suPAErnova.steps.pae import PAEStep
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.model import AbstractModelStepConfig

from .model import NFlowModelConfig


class NFlowStepConfig[Backend: str](AbstractModelStepConfig[Backend, NFlowModelConfig]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, Callable[[], type[NFlowModelConfig]]]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFNFlowModelConfig,
        "PyTorch": lambda: importlib.import_module(
            ".tch", __package__
        ).TCHNFlowModelConfig,
    }
    id: ClassVar[str] = "nflow"
    required_steps: ClassVar[list[str]] = [PAEStepConfig.id]

    # --- Previous Steps ---
    pae: PAEStep[Any] | None = None

    # --- Optional ---
    seed: int = 12345


NFlowStepConfig.register_step()
