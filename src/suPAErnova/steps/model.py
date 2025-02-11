from typing import TYPE_CHECKING, final, override

from suPAErnova.steps import Step
from suPAErnova.config.model import prev, optional, required

if TYPE_CHECKING:
    from suPAErnova.steps import REQ
    from suPAErnova.utils.typing import CFG


class Model(Step):
    required: list["REQ"] = required
    optional = optional
    prev = prev

    def __init__(self, cfg: "CFG") -> None:
        super().__init__(cfg)

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
