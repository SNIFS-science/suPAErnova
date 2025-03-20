# Copyright 2025 Patrick Armstrong
"""Global configurations shared by all steps in SuPAErnova."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from suPAErnova.typings.common import Input


class GlobalConfig(BaseModel):
    """Global configurations shared by all steps in SuPAErnova.

    Attributes:
        verbose (bool): Controls the log verbosity. If True, debug log messages will be printed to both stdout and the log file, and tqdm-based progress bars will be enabled. If False, only info log messages will be printed to stdout (though debug log messages will still be included in the log file) and progress bars will be disabled. Defaults to False
        force (bool): Force every step to run, ignoring previous results. If True, all steps will run, overwriting previous results. If False, each step will only run if previous results cannot be found and loaded instead. Defaults to False.
    """

    verbose: bool
    force: bool

    @staticmethod
    def init(
        input_config: "Input",
        *,  # Force keyword-only arguments
        verbose: bool = False,
        force: bool = False,
    ) -> "GlobalConfig":
        """Setup global configurations used by all steps.

        Args:
            input_config (Input): Configuration dictionary to setup / update.

        Kwargs:
            verbose (bool): Increase log verbosity. Defaults to False.
            force (bool): Force every step to run, even if a previous result already exists. Defaults to False.

        Returns:
            GlobalConfig: The global configuration.
        """
        default_global_config = {"verbose": verbose, "force": force}
        global_config = {**default_global_config, **input_config}
        return GlobalConfig.model_validate(global_config)
