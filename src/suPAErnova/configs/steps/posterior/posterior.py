# Copyright 2025 Patrick Armstrong

from typing import Any, ClassVar, get_args

from pydantic import Field, model_validator

from suPAErnova.steps.nflow import NFlowStep
from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.pae.model import TFBackend, TCHBackend

from .tf import TFPosteriorModelConfig
from .tch import TCHPosteriorModelConfig

ModelConfig = TFPosteriorModelConfig | TCHPosteriorModelConfig


class PosteriorStepConfig(StepConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "posterior"
    required_steps: ClassVar[list[str]] = [NFlowStepConfig.id]

    # --- Previous Steps ---
    nflow: NFlowStep | None = None

    # --- Models ---
    model: ModelConfig
    models: list[ModelConfig] = Field(validation_alias="variant")

    @model_validator(mode="before")
    @classmethod
    def prep_model_config(cls, data: Any) -> Any:
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

            posterior_model_configs = [
                data["model"],
                *[
                    {**base_model_config, **model_config}
                    for model_config in data.get("variant", [])
                ],
            ]

            data.pop("variant", None)
            data["variant"] = []
            for i, posterior_model_config in enumerate(posterior_model_configs):
                backend = posterior_model_config.get("backend")
                if backend is None:
                    err = f"{'Base' if i == 0 else f'Variant {i}'} Model is missing a backend key. Please choose from {get_args(TFBackend)} for TensorFlow or {get_args(TCHBackend)} for PyTorch"
                    raise ValueError(err)
                if backend in get_args(TFBackend):
                    model_config_cls = TFPosteriorModelConfig
                elif posterior_model_config.get("backend") in get_args(TCHBackend):
                    model_config_cls = TCHPosteriorModelConfig
                else:
                    err = f"Unknown backend: {backend}. Please choose from {get_args(TFBackend)} for TensorFlow or {get_args(TCHBackend)} for PyTorch"
                    raise ValueError(err)
                model_config = model_config_cls.from_config(posterior_model_config)
                data["variant"].append(model_config)
        return data

    # --- Optional ---
    seed: int = 12345


PosteriorStepConfig.register_step()
