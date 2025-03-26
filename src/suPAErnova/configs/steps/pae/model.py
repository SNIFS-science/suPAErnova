# Copyright 2025 Patrick Armstrong

from typing import Self, Literal, ClassVar
from pathlib import Path  # noqa: TC003
import itertools

from pydantic import (
    PositiveInt,  # noqa: TC002
    field_validator,
    model_validator,
)

from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.steps.data import DataStepConfig


class PAEModelConfig(StepConfig):
    # --- Class Variables ---
    name: ClassVar["str"] = "pae_model"
    required_steps: ClassVar["list[str]"] = [DataStepConfig.name]

    # --- Required ---
    architecture: Literal["dense", "convolutional"]
    encode_dims: list["PositiveInt"]

    # --- Optional ---
    colourlaw: "Path | None" = None

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.colourlaw is not None:
            self.colourlaw = self.paths.resolve_path(
                self.colourlaw, relative_path=self.paths.base
            )
            if not self.colourlaw.exists():
                err = f"`colourlaw` resolved to {self.colourlaw}, which does not exist."
                raise ValueError(err)
        return self

    @field_validator("encode_dims", mode="before")
    @classmethod
    def validate_encode_dims(cls, value: list[int]) -> list[int]:
        if len(value) == 0:
            err = "`encode_dims` can not be empty"
            raise ValueError(err)
        if not all(x > y for x, y in itertools.pairwise(value)):
            err = f"`encode_dims`: {value} is not monotonically decreasing"
            raise ValueError(err)
        return value
