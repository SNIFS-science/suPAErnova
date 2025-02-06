from collections.abc import Iterable
from typing import Any


type CONFIG[T] = dict[str, T]
type CONFIG_DATA = str | int | bool | Iterable[CONFIG_DATA] | CONFIG[CONFIG_DATA]
type INPUT = CONFIG[CONFIG_DATA]
type CFG = CONFIG[Any]
