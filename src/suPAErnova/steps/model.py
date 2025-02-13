from typing import TYPE_CHECKING, TypeVar, ClassVar, override

from suPAErnova.steps import Step, callback
from suPAErnova.models import models
from suPAErnova.config.model import (
    prev,
    optional,
    required,
    optional_params,
    required_params,
)

if TYPE_CHECKING:
    from suPAErnova.steps.data import Data
    from suPAErnova.utils.typing import CFG
    from suPAErnova.config.requirements import REQ

    M = TypeVar("M")


class ModelStep(Step):
    required = required
    optional = optional
    prev = prev
    required_params: ClassVar[list["REQ"]] = required_params
    optional_params: ClassVar[list["REQ"]] = optional_params

    @classmethod
    def __update_params__(cls) -> None:
        if cls is ModelStep:
            return
        parent_cls = next(
            (
                parent
                for parent in cls.__bases__
                if issubclass(parent, ModelStep) and parent is not ModelStep
            ),
            ModelStep,
        )
        # Create new lists instead of mutating to avoid side effects
        cls.required_params = parent_cls.required_params + cls.required_params
        cls.optional_params = parent_cls.optional_params + cls.optional_params

    def __init_subclass__(cls, **kwargs: "CFG") -> None:
        super().__init_subclass__(**kwargs)
        cls.__update_params__()

    def __init__(self, cfg: "CFG") -> None:
        super().__init__(cfg)
        self.data: Data = self.global_cfg["RESULTS"]["DATA"]
        self.params: CFG = self.opts["PARAMS"]

        is_valid = self.validate_params()
        if not is_valid:
            msg = f"Invalid {self.name} configuration"
            raise ValueError(msg)

        self.model: M
        self.model_cls: type[M]

    @override
    def _setup(self):
        model_cls = self.opts["MODEL"]
        if model_cls is None:
            name = self.__class__.__name__.upper()
            model_cls = models.get(name)
            if model_cls is None:
                return False, f"Unknown Model: {name}, must be one of {models}"
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
    def _load(self) -> None:
        return None

    @override
    def _run(self):
        return True, None

    @override
    def _result(self):
        return True, None
