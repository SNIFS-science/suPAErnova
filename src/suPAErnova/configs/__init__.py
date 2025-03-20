# Copyright 2025 Patrick Armstrong
"""Pydantic Models for use throughout SuPAErnova."""

from typing import TYPE_CHECKING

# Used to define Pydantic model, so don't put into TYPE_CHECKING block
from logging import Logger  # NOQA: TC003

import toml
from pydantic import BaseModel, ConfigDict

from suPAErnova.logging import setup_logging

# Used to define Pydantic model, so don't put into TYPE_CHECKING block
from suPAErnova.configs.paths import PathConfig  # NOQA: TC001
from suPAErnova.configs.globals import GlobalConfig  # NOQA: TC001

if TYPE_CHECKING:
    from suPAErnova.typings import Input, Configuration


class SNPAEConfig(BaseModel):
    """Generic model from which most SuPAErnova configs inherit."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)  # pyright: ignore[reportIncompatibleVariableOverride]

    globals: "GlobalConfig"
    paths: "PathConfig"
    log: "Logger"

    @classmethod
    def init(
        cls,
        input_config: "Configuration",
    ) -> "SNPAEConfig":
        """User-defined input configuration which controls the behaviour of SuPAErnova.

        Args:
            input_config (Configuration): User-defined input configuration.

        Returns:
            InputConfig: The input configuration.
        """
        default_config = {
            "log": setup_logging(
                cls.__name__,
                log_path=input_config["paths"].log,
                verbose=input_config["globals"].verbose,
            )
        }
        config = {**default_config, **input_config}
        cfg = cls.model_validate(config)
        cfg.log.debug(f"{cls.__name__}:\n{cfg.model_dump()}")
        cfg.save()
        return cfg

    def save(self) -> None:
        """Save config file to `self.paths.out / f"{self.__class__.__class__}.toml"."""
        save_file = self.paths.out / f"{self.__class__.__name__}.toml"
        with save_file.open(
            "w",
            encoding="utf-8",
        ) as io:
            toml.dump(self.model_dump(exclude={"log"}), io)

    @staticmethod
    def normalise_input(input_config: "Input") -> "Input":
        """Prepare an input config for use by forcing all keys to be lowercase.

        Args:
            input_config (Input): The input config to normalise.

        Returns:
            Input: The normalised config.
        """
        rtn: Input = {}
        for k, v in input_config.items():
            val = v
            if isinstance(v, dict):
                val = SNPAEConfig.normalise_input(v)
            rtn[k.lower()] = val
        return rtn
