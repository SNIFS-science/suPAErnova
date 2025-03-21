# Copyright 2025 Patrick Armstrong


from typing import (
    Self,
    ClassVar,
    Annotated,
    cast,
)
from pathlib import Path

from astropy import cosmology as cosmo
from pydantic import (
    Field,
    PositiveInt,  # noqa: TC002
    field_validator,
    model_validator,
)

from suPAErnova.configs.steps import StepConfig


class DataStepConfig(StepConfig):
    # Class Vars
    name: ClassVar["str"] = "data"

    # Required
    data_dir: "Path"
    meta: "Path"
    idr: "Path"
    mask: "Path"

    # Optional
    cosmological_model: str = "WMAP7"
    salt_model: "str | Path" = "salt2"
    min_phase: float = -10
    max_phase: float = 40
    train_frac: Annotated[float, Field(ge=0, le=1)] = 0.75
    seed: PositiveInt

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        self.data_dir = self.paths.resolve_path(
            self.data_dir, relative_path=self.paths.base
        )
        if not self.data_dir.exists():
            err = f"`data_dir` resolved to {self.data_dir}, which does not exist."
            raise ValueError(err)

        for field, ext in {"meta": ".csv", "idr": ".txt", "mask": ".txt"}.items():
            setattr(
                self,
                field,
                self.paths.resolve_path(
                    getattr(self, field), relative_path=self.data_dir
                ),
            )

            field_path = cast("Path", getattr(self, field))

            if not field_path.exists():
                err = f"`{field}` resolved to {field_path}, which does not exist."
                raise ValueError(err)

            if field_path.suffix != ext:
                err = f"`{field}` resolved to {field_path}, which is not a {ext} file."
                raise ValueError(err)

        return self

    @field_validator("cosmological_model", mode="after")
    @classmethod
    def validate_cosmological_model(cls, value: str) -> str:
        if value not in cosmo.realizations.available:
            err = f"`cosmological_model` is {value} but must be one of {cosmo.realizations.available}"
            raise ValueError(err)
        return value

    @field_validator("salt_model", mode="after")
    @classmethod
    def validate_salt_model(cls, value: str) -> str:
        if ("salt2" not in value) and ("salt3" not in value):
            err = f'`salt_model` is {value} but does not appear to be a salt2 or salt3 model, as it does not contain the string `"salt2"` or `"salt3"'
            raise ValueError(err)
        return value

    @model_validator(mode="after")
    def validate_salt_model_path(self) -> Self:
        salt_path = self.paths.resolve_path(
            Path(self.salt_model), relative_path=self.paths.base
        )
        if salt_path.exists():
            self.salt_model = salt_path
        return self

    @model_validator(mode="after")
    def validate_max_phase(self) -> Self:
        if self.max_phase <= self.min_phase:
            err = f"`max_phase`: {self.max_phase} is not strictly greater than `min_phase`: {self.min_phase}"
            raise ValueError(err)
        return self


DataStepConfig.register_step()
