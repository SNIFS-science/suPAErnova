# Copyright 2025 Patrick Armstrong
"""Global configurations shared by all steps in SuPAErnova."""

from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    StrictBool,  # noqa: TC002
)

if TYPE_CHECKING:
    from pydantic import JsonValue


class GlobalConfig(BaseModel):
    verbose: "StrictBool"
    force: "StrictBool"

    @classmethod
    def from_config(
        cls,
        input_config: dict[str, "JsonValue"],
        *,  # Force keyword-only arguments
        verbose: "StrictBool" = False,
        force: "StrictBool" = False,
    ) -> "GlobalConfig":
        config = {**cls.default_config(verbose=verbose, force=force), **input_config}
        return cls.model_validate(config)

    @classmethod
    def default_config(
        cls,
        *,  # Force keyword-only arguments
        verbose: "StrictBool" = False,
        force: "StrictBool" = False,
    ) -> dict[str, "Any"]:
        return {"verbose": verbose, "force": force}
