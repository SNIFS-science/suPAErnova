# Copyright 2025 Patrick Armstrong
from typing import Any, Literal, ClassVar, get_args

from pydantic import model_validator

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.pae.tf.model import TFPAEModelConfig
from suPAErnova.configs.steps.pae.tch.model import TCHPAEModelConfig

TFBackend = Literal["tf", "tensorflow"]
TCHBackend = Literal["tch", "torch"]
Backend = TFBackend | TCHBackend

ModelConfig = TFPAEModelConfig | TCHPAEModelConfig


class PAEStepConfig(StepConfig):
    # --- Class Variables ---
    name: ClassVar[str] = "pae"
    required_steps: ClassVar[list[str]] = [DataStepConfig.name]

    # --- Required ---
    backend: "Backend"
    model: "ModelConfig"

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
            pae_model_config = {**model_config, **data.get("model", {})}
            data.pop("model")
            if data["backend"] in get_args(TFBackend):
                data["model"] = TFPAEModelConfig.from_config(pae_model_config)
            else:
                data["model"] = TCHPAEModelConfig.from_config(pae_model_config)
        return data


PAEStepConfig.register_step()
