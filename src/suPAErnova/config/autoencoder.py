from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from suPAErnova.steps import REQ


required: list["REQ"] = []
optional: list["REQ"] = []
prev: list[str] = []
