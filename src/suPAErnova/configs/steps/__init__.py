# Copyright 2025 Patrick Armstrong

from copy import deepcopy
from typing import Any, ClassVar, cast, override

from suPAErnova.configs import SNPAEConfig
from suPAErnova.configs.paths import PathConfig


class StepConfig(SNPAEConfig):
    # Class Variables
    steps: ClassVar["dict[str, type[StepConfig]]"] = {}
    required_steps: ClassVar["list[str]"] = []
    name: ClassVar["str"] = "step"

    @classmethod
    def register_step(cls) -> None:
        cls.steps[cls.name] = cls

    @override
    @classmethod
    def from_config(
        cls,
        input_config: dict[str, "Any"],
    ) -> "StepConfig":
        step_config = deepcopy(input_config)
        step_config["paths"].out = PathConfig.resolve_path(
            step_config["paths"].out / cls.__name__,
            relative_path=step_config["paths"].base,
            mkdir=True,
        )
        step_config["paths"].log = PathConfig.resolve_path(
            step_config["paths"].out / "logs",
            relative_path=step_config["paths"].base,
            mkdir=True,
        )
        return cast("StepConfig", super().from_config(step_config))
