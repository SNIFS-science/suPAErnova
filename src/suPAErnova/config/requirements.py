# Copyright 2025 Patrick Armstrong
"""Define configuration specification, with the option to enforce types, bounds, and arbitrary function."""

import typing
from typing import TYPE_CHECKING, cast, final
import traceback

from suPAErnova.utils import suPAErnova_logging as log

if TYPE_CHECKING:
    from typing import Any, TypeVar
    from collections.abc import Callable, Collection

    from suPAErnova.utils.suPAErnova_types import CFG, CONFIG_DATA

    # --- Types ---
    type RequirementReturn[T] = tuple[bool, T | str]
    IN = TypeVar("IN", bound=CONFIG_DATA)
    OUT = TypeVar("OUT", bound=Any)

    # TODO: Add more restrictions
    #       positive / strictly positive
    #       negative / strictly negative
    #       min but not max
    #       max but not min
    #       max with min of 0

    # TODO: Add some default transforms
    #       int | float -> int


@final
class Requirement[IN, OUT]:
    """A configuration requirement which takes in an input: IN, transforms it into an output: OUT, and checks whether the output is valid."""

    def __init__(
        self,
        name: str,
        description: str,
        default: IN | None = None,
        choice: "Collection[IN] | None" = None,
        bounds: tuple[IN, IN] | None = None,
        transform: "Callable[[IN, CFG, CFG], RequirementReturn[OUT]] | None" = None,
    ) -> None:
        """Initialise Requirement."""
        self.name = name
        self.description = description
        self.default = default
        self.choice = choice
        self.bounds = bounds
        self.transform = transform

    def __in_type(self) -> type[IN]:
        """Get the runtime type associated with IN.

        Returns:
            type[IN]
        """
        return typing.get_args(self.__orig_class__)[0]  # pyright:ignore[reportAttributeAccessIssue]

    def __out_type(self) -> type[OUT]:
        """Get the runtime type associated with OUT.

        Returns:
            type[OUT]
        """
        return typing.get_args(self.__orig_class__)[-1]  # pyright:ignore[reportAttributeAccessIssue]

    def validate_type(self, opt: object) -> "RequirementReturn[IN]":
        """Validate that the user has passed the correct input type for this requirement.

        Args:
            opt (object): The input provided by the user

        Returns:
            RequirementReturn[IN]
        """
        if not isinstance(opt, self.__in_type()):
            return (False, f"Incorrect type {type(opt)}, must be {self.__in_type()}")
        return True, opt

    def validate_choice(self, opt: IN) -> "RequirementReturn[IN]":
        """Validate that the user has chosen one of the available choices for this requirement.

        Args:
            opt (IN): The input provided by the user

        Returns:
            RequirementReturn[IN]
        """
        if self.choice is not None and opt not in self.choice:
            return (False, f"Unknown choice {opt}, must be one of {self.choice}")
        return True, opt

    def validate_bounds(self, opt: IN) -> "RequirementReturn[IN]":
        """Validate that the user has provided an input within the bounds of this requirement.

        Args:
            opt (IN): The input provided by the user

        Returns:
            RequirementReturn[IN]
        """
        if self.bounds is not None:
            try:
                if not (self.bounds[0] <= opt <= self.bounds[-1]):  # pyright:ignore[reportOperatorIssue]
                    return (False, f"{opt} must be within {self.bounds}")
            except TypeError:
                return (
                    False,
                    f"operator <= is not supported for {opt}: {self.__in_type}",
                )
        return True, opt

    def validate_transform(
        self,
        opt: IN,
        cfg: "CFG",
        opts: "CFG",
    ) -> "RequirementReturn[OUT]":
        """Transform the input into the desired output type, and validate that the transformation was successful.

        Args:
            opt (IN): The input provided by the user
            cfg (CFG): Global config
            opts (CFG): Transformation specific config

        Returns:
            RequirementReturn[OUT]
        """
        if self.transform is not None:
            try:
                return self.transform(opt, cfg, opts)
            except Exception:
                result = f"Error tranforming {opt}: {traceback.format_exc()}"
                log.exception(result)
                return False, result
        return True, cast("OUT", opt)

    def validate(self, opt: IN, cfg: "CFG", opts: "CFG") -> "RequirementReturn[OUT]":
        """Validate user input for this requirement.

        Args:
            opt (IN): The input provided by the user
            cfg (CFG): Global config
            opts (CFG): Transformation specific config

        Returns:
            RequirementReturn[OUT]
        """
        ok, result = self.validate_type(opt)
        if ok:
            ok, result = self.validate_choice(opt)
        if ok:
            ok, result = self.validate_bounds(opt)
        if ok:
            ok, result = self.validate_transform(opt, cfg, opts)
        result = cast("str", result) if not ok else cast("OUT", result)
        return ok, result


if TYPE_CHECKING:
    type REQ = Requirement[Any, Any]
