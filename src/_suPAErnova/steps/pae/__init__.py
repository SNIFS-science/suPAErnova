from typing import TYPE_CHECKING, TypeVar, ClassVar, override

from suPAErnova.steps import Step_contra, callback
from suPAErnova.config.pae import (
    prev,
    optional,
    required,
    optional_params,
    required_params,
)
from suPAErnova.models.pae import models

if TYPE_CHECKING:
    from suPAErnova.steps.data import DATAStep
    from suPAErnova.config.requirements import REQ, RequirementReturn
    from suPAErnova.utils.suPAErnova_types import CFG

    M = TypeVar("M")


class PAEStep(Step_contra):
    required = required
    optional = optional
    prev = prev
    required_params: ClassVar[list["REQ"]] = required_params
    optional_params: ClassVar[list["REQ"]] = optional_params

    @classmethod
    def __update_params__(cls) -> None:
        if cls is PAEStep:
            return
        parent_cls = next(
            (
                parent
                for parent in cls.__bases__
                if issubclass(parent, PAEStep) and parent is not PAEStep
            ),
            PAEStep,
        )
        # Create new lists instead of mutating to avoid side effects
        cls.required_params = parent_cls.required_params + cls.required_params
        cls.optional_params = parent_cls.optional_params + cls.optional_params

    def __init_subclass__(cls, **kwargs: "CFG") -> None:
        super().__init_subclass__(**kwargs)
        cls.__update_params__()

    def __init__(self, cfg: "CFG") -> None:
        super().__init__(cfg)
        self.data: DATAStep = self.global_cfg["RESULTS"]["DATA"]
        self.params: CFG = self.opts["PARAMS"]

        is_valid = self.validate_params()
        if not is_valid:
            msg = f"Invalid {self.name} configuration"
            raise ValueError(msg)

        self.model: M
        self.model_cls: type[M]

    @override
    def _setup(self):
        super()._setup()
        model_cls = self.opts["MODEL"]
        if model_cls is None:
            model_name = self.name.upper()
            model_cls = models.get(model_name)
            if model_cls is None:
                return False, f"Unknown Model: {model_name}, must be one of {models}"
        self.model_cls = model_cls
        return (True, None)

    @callback
    def validate_params(self) -> bool:
        for requirement in self.required_params:
            key = requirement.name.upper()
            opt = self.params.get(key)
            if opt is None:
                self.log.error(
                    f"{self.name} is missing required option: {key}: {requirement.description}",
                )
                return False
            ok, result = requirement.validate(opt, self.global_cfg, self.params)
            if not ok:
                self.log.error(f"Invalid `{key}`=`{opt}`: {result}")
                return False
            self.params[key] = result

        for requirement in self.optional_params:
            key = requirement.name.upper()
            opt = self.params.get(key)
            if opt is None:
                opt = requirement.default
            if opt is not None:
                ok, result = requirement.validate(opt, self.global_cfg, self.params)
                if not ok:
                    self.log.error(f"Invalid `{key}`=`{opt}`: {result}")
                    return False
            else:
                result = None
            self.params[key] = result
        return True

    @override
    def _is_completed(self) -> bool:
        return False

    @override
    def _load(self) -> "RequirementReturn[None]":
        return True, None

    @override
    def _run(self):
        return True, None

    @override
    def _result(self):
        return True, None
