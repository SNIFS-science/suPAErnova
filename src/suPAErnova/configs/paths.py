# Copyright 2025 Patrick Armstrong
"""Paths used throughout SuPAErnova."""

from typing import TYPE_CHECKING
from pathlib import Path

from pydantic import BaseModel

if TYPE_CHECKING:
    from suPAErnova.typings import Input


class PathConfig(BaseModel):
    """Paths used throughout SuPAErnova.

    Attributes:
        base (Path): The directory from which all paths are assumed to be relative. Defaults to the parent directory of input_path if using the cli, otherwise defaults to cwd.
        out (Path): Path that will contain all SuPAErnova output files. Defaults to `base/output`
        log (Path): Path to log file. Defaults to `base/__name__.log`
    """

    base: Path = Path.cwd()
    out: Path
    log: Path

    @staticmethod
    def init(
        input_config: "Input",
        *,  # Force keyword-only arguments
        base_path: "Path",
        out_path: "Path",
        log_path: "Path",
    ) -> "PathConfig":
        """Setup paths used by all steps.

        Args:
            input_config (Input): Configuration dictionary to setup / update.

        Kwargs:
            base_path (Path): The directory from which all paths are assumed to be relative.
            out_path (Path): Directory that will contain all SuPAErnova output files.
            log_path (Path): Directory that will contain all log files.

        Returns:
            GlobalConfig: The global configuration.
        """
        default_path_config = {"base": base_path, "out": out_path, "log": log_path}
        path_config = {**default_path_config, **input_config}
        return PathConfig.model_validate(path_config)

    @staticmethod
    def resolve_path(
        input_path: Path | None,
        *,
        default_path: Path,
        relative_path: Path,
        mkdir: bool = False,
    ) -> Path:
        """Resolve path to be absolute.

        Args:
            input_path (Path | None): The path to resolve, or None to use the default_path.
            default_path (Path): The path to default back to if no input_path is provided.
            relative_path (Path): The path that input_path should be relative to, if an absolute path was not provided.
            mkdir (Bool): Whether the resolved path should be created if it doesn't exist. Defaults to False.

        Returns:
            Path: The resolved path
        """
        if input_path is None:
            input_path = default_path
        if not input_path.is_absolute():
            input_path = relative_path / input_path
        final_path = input_path.resolve()
        if mkdir:
            final_path.mkdir(parents=True, exist_ok=True)
        return final_path
