import typing
from typing import TYPE_CHECKING, TypeVar, cast, final
import traceback

if TYPE_CHECKING:
    from typing import Any, Literal
    from collections.abc import Callable, Collection

    from suPAErnova.utils.typing import CFG

# --- Types ---
type RequirementReturn[T] = tuple[Literal[False], str] | tuple[Literal[True], T]
IN = TypeVar("IN")
OUT = TypeVar("OUT")


@final
class Requirement[IN, OUT]:
    def __init__(
        self,
        name: str,
        description: str,
        default: IN | None = None,
        choice: "Collection[IN] | None" = None,
        bounds: tuple[IN, IN] | None = None,
        transform: "Callable[[IN, CFG, CFG], RequirementReturn[IN | OUT]] | None" = None,
    ) -> None:
        self.name = name
        self.description = description
        self.default = default
        self.choice = choice
        self.bounds = bounds
        self.transform = transform

    def _IN(self):
        return typing.get_args(self.__orig_class__)[0]

    def _OUT(self):
        return typing.get_args(self.__orig_class__)[-1]

    def validate_type(self, opt: object) -> RequirementReturn[IN]:
        if not isinstance(opt, self._IN()):
            return (False, f"Incorrect type {type(opt)}, must be {self._IN()}")
        # If isinstance(opt, IN), cast to IN
        return True, cast("IN", opt)

    def validate_choice(self, opt: IN) -> RequirementReturn[IN]:
        if self.choice is not None and opt not in self.choice:
            return (False, f"Unknown choice {opt}, must be one of {self.choice}")
        return True, opt

    def validate_bounds(self, opt: IN) -> RequirementReturn[IN]:
        if self.bounds is not None and not (self.bounds[0] <= opt <= self.bounds[-1]):
            return (False, f"{opt} must be within {self.bounds}")
        return True, opt

    def validate_transform(
        self,
        opt: IN,
        cfg: "CFG",
        opts: "CFG",
    ) -> RequirementReturn[IN | OUT]:
        if self.transform is not None:
            try:
                return self.transform(opt, cfg, opts)
            except Exception:
                return False, f"Error tranforming {opt}: {traceback.format_exc()}"
        return True, opt

    def validate(self, opt: IN, cfg: "CFG", opts: "CFG"):
        ok, result = self.validate_type(opt)
        if not ok:
            return ok, result
        ok, result = self.validate_choice(opt)
        if not ok:
            return ok, result
        ok, result = self.validate_bounds(opt)
        if not ok:
            return ok, result
        ok, result = self.validate_transform(opt, cfg, opts)
        if not ok:
            return ok, result
        return (ok, result)


type REQ = Requirement[Any, Any]
