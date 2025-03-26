# Copyright 2025 Patrick Armstrong
from typing import Any, Literal, ClassVar

from pydantic import model_validator

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.pae.model import PAEModelConfig  # noqa: TC001

TFBackend = Literal["tf", "tensorflow"]
TCHBackend = Literal["tch", "torch"]
Backend = TFBackend | TCHBackend


class PAEStepConfig(StepConfig):
    # --- Class Variables ---
    name: ClassVar[str] = "pae"
    required_steps: ClassVar[list[str]] = [DataStepConfig.name]

    # --- Required ---
    backend: "Backend"
    pae_model_config: "PAEModelConfig"

    # --- Optional ---

    @model_validator(mode="before")
    @classmethod
    def prep_model_config(cls, data: "Any") -> "Any":
        if isinstance(data, dict):
            model_config = {
                "paths": data.get("paths"),
                "config": data.get("config"),
                "log": data.get("log"),
            }
            data["pae_model_config"] = {**model_config, **data.get("model", {})}
            data.pop("model")
        return data


PAEStepConfig.register_step()
