# Copyright 2025 Patrick Armstrong
from typing import Any, ClassVar, Annotated, get_args

from pydantic import (
    Field,
    model_validator,
)

from suPAErnova.steps.data import DataStep
from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig
from suPAErnova.configs.steps.pae.tch import TCHPAEModelConfig
from suPAErnova.configs.steps.pae.model import TFBackend

ModelConfig = TFPAEModelConfig | TCHPAEModelConfig


class PAEStepConfig(StepConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "pae"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id]

    # --- Previous Steps ---
    data: DataStep | None = None
    validation_frac: Annotated[float, Field(ge=0, le=0)]

    # --- Models ---
    model: ModelConfig
    models: list["ModelConfig"] = Field(validation_alias="variant")

    @model_validator(mode="before")
    @classmethod
    def prep_model_config(cls, data: "Any") -> "Any":
        if isinstance(data, dict):
            default_model_config = {
                "paths": data.get("paths"),
                "config": data.get("config"),
                "log": data.get("log"),
            }
            base_model_config = {**default_model_config, **data.get("model", {})}
            data["model"] = base_model_config

            pae_model_configs = [data["model"]] + [
                {**base_model_config, **model_config}
                for model_config in data.get("variant", [])
            ]
            data.pop("variant")
            data["variant"] = [
                TFPAEModelConfig.from_config(pae_model_config)
                if pae_model_config["backend"] in get_args(TFBackend)
                else TCHPAEModelConfig.from_config(pae_model_config)
                for pae_model_config in pae_model_configs
            ]
        return data


PAEStepConfig.register_step()
