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
from suPAErnova.configs.steps.pae.model import TFBackend, TCHBackend

ModelConfig = TFPAEModelConfig | TCHPAEModelConfig


class PAEStepConfig(StepConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "pae"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id]

    # --- Previous Steps ---
    data: DataStep | None = None
    validation_frac: Annotated[float, Field(ge=0, le=1)]

    # --- Models ---
    model: ModelConfig
    models: list["ModelConfig"] = Field(validation_alias="variant")

    @model_validator(mode="before")
    @classmethod
    def prep_model_config(cls, data: "Any") -> "Any":
        if isinstance(data, dict):
            if "model" not in data:
                err = f"No Base Model has been defined. Please define one in [{cls.id}.model]"
                raise ValueError(err)

            default_model_config = {
                "paths": data.get("paths"),
                "config": data.get("config"),
                "log": data.get("log"),
            }
            base_model_config = {**default_model_config, **data.get("model", {})}
            data["model"] = base_model_config

            pae_model_configs = [
                data["model"],
                *[
                    {**base_model_config, **model_config}
                    for model_config in data.get("variant", [])
                ],
            ]
            data.pop("variant", None)
            data["variant"] = []
            for i, pae_model_config in enumerate(pae_model_configs):
                backend = pae_model_config.get("backend")
                if backend is None:
                    err = f"{'Base' if i == 0 else f'Variant {i}'} Model is missing a backend key. Please choose from {get_args(TFBackend)} for TensorFlow or {get_args(TCHBackend)} for PyTorch"
                    raise ValueError(err)
                if backend in get_args(TFBackend):
                    model_config_cls = TFPAEModelConfig
                elif pae_model_config.get("backend") in get_args(TCHBackend):
                    model_config_cls = TCHPAEModelConfig
                else:
                    err = f"Unknown backend: {backend}. Please choose from {get_args(TFBackend)} for TensorFlow or {get_args(TCHBackend)} for PyTorch"
                    raise ValueError(err)
                model_config = model_config_cls.from_config(pae_model_config)
                data["variant"].append(model_config)
        return data

    # --- Optional ---
    seed: int = 12345


PAEStepConfig.register_step()
