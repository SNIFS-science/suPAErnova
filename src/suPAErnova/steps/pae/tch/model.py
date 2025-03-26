# Copyright 2025 Patrick Armstrong

from typing import Any

from pydantic import BaseModel


class TCHPAEModelConfig(BaseModel):
    pass


class TCHPAEModel:
    def __init__(
        self,
        config_dict: dict[str, Any],
    ) -> None:
        config = TCHPAEModelConfig.model_validate(config_dict)
