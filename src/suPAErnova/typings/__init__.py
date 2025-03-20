# Copyright 2025 Patrick Armstrong
"""Utility types used throughout SuPAErnova."""

from typing import Any
from collections.abc import Sequence, MutableMapping

type Options[V] = MutableMapping[str, V]
type InputValue = str | int | float | Sequence[InputValue] | Options[InputValue]
type Input = Options[InputValue]
type Configuration = Options[Any]
