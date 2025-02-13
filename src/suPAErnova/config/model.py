from typing import TYPE_CHECKING

from suPAErnova.models import PAEModel, models
from suPAErnova.config.requirements import Requirement

if TYPE_CHECKING:
    from suPAErnova.utils.typing import CFG
    from suPAErnova.config.requirements import REQ, RequirementReturn


def get_model(
    name: str,
    _1: "CFG",
    _2: "CFG",
) -> "RequirementReturn[type[PAEModel] | None]":
    if not name:
        return (True, None)
    model = models.get(name.upper())
    if model is None:
        return False, f"Unknown Model: {name}, must be one of {models}"
    return True, model


model = Requirement[str, type[PAEModel] | None](
    name="model",
    description=" The type of model to use",
    default="",
    transform=get_model,
)

params = Requirement[dict, dict](
    name="params",
    description="Parameters needed to setup and run model",
    default={},
)


required: list["REQ"] = []
optional: list["REQ"] = [model, params]
prev: list[str] = ["DATA"]
required_params: list["REQ"] = []
optional_params: list["REQ"] = []
