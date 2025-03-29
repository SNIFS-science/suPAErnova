# Copyright 2025 Patrick Armstrong

from typing import Any, Self, Protocol
from logging import Logger
from pathlib import Path
from collections.abc import Callable

import toml
from pydantic import BaseModel, ConfigDict, model_validator

from suPAErnova.logging import setup_logging
from suPAErnova.configs.paths import PathConfig
from suPAErnova.configs.config import GlobalConfig


class CallbackFunc[Instance: "Any", Returns](Protocol):
    def __call__(_self, self: Instance, *args: "Any", **kwargs: "Any") -> Returns: ...


def callback[Instance: "Any", Returns](
    fn: CallbackFunc[Instance, Returns],
) -> "Callable[..., Returns]":
    def wrapper(self: "Instance", *args: "Any", **kwargs: "Any") -> "Returns":
        callbacks: dict[str, Callable[[Instance], None]] = self.options.callbacks.get(
            fn.__name__.lower(), {}
        )
        pre_callback = callbacks.get("pre")
        if pre_callback is not None:
            pre_callback(self)
        rtn = fn(self, *args, **kwargs)
        post_callback = callbacks.get("post")
        if post_callback is not None:
            post_callback(self)
        return rtn

    return wrapper


class SNPAEConfig(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # pyright: ignore[reportIncompatibleVariableOverride]

    # Required
    config: "GlobalConfig"
    paths: "PathConfig"
    log: "Logger"

    # Optional
    callbacks: dict[str, str | dict[str, "Callable[[Any], None]"]] = {}

    @model_validator(mode="after")
    def validate_callbacks(self) -> "Self":
        for fn, callback in self.callbacks.items():
            if isinstance(callback, str):
                fn_callbacks = {}
                callback_path = self.paths.resolve_path(
                    Path(callback), relative_path=self.paths.base
                )
                if not callback_path.exists():
                    err = f"`{fn}` callback: `{callback}` resolved to `{callback_path}`, which does not exist."
                    raise ValueError(err)
                with callback_path.open("r") as io:
                    script_code = io.read()
                # Create an isolated namespace
                local_scope = {}
                exec(script_code, globals(), local_scope)  # noqa: S102
                if "pre" in local_scope:
                    if not isinstance(local_scope["pre"], Callable):
                        err = f"`pre-{fn}` callback is not callable"
                        raise ValueError(err)
                    fn_callbacks["pre"] = local_scope["pre"]
                if "post" in local_scope:
                    if not isinstance(local_scope["post"], Callable):
                        err = f"`pre-{fn}` callback is not callable"
                        raise ValueError(err)
                    fn_callbacks["post"] = local_scope["post"]
                self.callbacks[fn] = fn_callbacks
        self.callbacks = self.normalise_input(self.callbacks)
        return self

    @classmethod
    def from_config(
        cls,
        input_config: dict[str, "Any"],
    ) -> "Self":
        config = {**cls.default_config(input_config), **input_config}
        cfg = cls.model_validate(config)
        cfg.save()
        return cfg

    @classmethod
    def default_config(cls, input_config: dict[str, "Any"]) -> dict[str, "Any"]:
        return {
            "log": setup_logging(
                cls.__name__,
                log_path=input_config["paths"].log,
                verbose=input_config["config"].verbose,
            )
        }

    def save(self) -> None:
        save_file = self.paths.log / f"{self.__class__.__name__}.toml"
        with save_file.open(
            "w",
            encoding="utf-8",
        ) as io:
            toml.dump(self.model_dump(exclude={"log"}), io)

    @staticmethod
    def normalise_input(input_config: dict[str, "Any"]) -> dict[str, "Any"]:
        rtn: dict[str, Any] = {}
        for k, v in input_config.items():
            val = v
            if isinstance(v, dict):
                val = SNPAEConfig.normalise_input(v)
            rtn[k.lower()] = val
        return rtn
