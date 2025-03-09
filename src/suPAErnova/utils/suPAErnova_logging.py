# Copyright 2025 Patrick Armstrong
"""Logging utilities for SuPAErnova."""

from typing import TYPE_CHECKING
import logging

import coloredlogs

if TYPE_CHECKING:
    from pathlib import Path


def log(msg: str, level: int) -> None:
    """Convenvience logging function.

    Args:
        msg (str): Message to log
        level (int): Log level
    """
    logger = logging.getLogger()
    logger.log(level, msg)


def exception(msg: str) -> None:
    """Convenvience exception function.

    Args:
        msg (str): Exception message
    """
    logger = logging.getLogger()
    logger.exception(msg)  # noqa: LOG004


def error(msg: str) -> None:
    """Convenvience error logging function.

    Args:
        msg (str): Error message
    """
    log(msg, logging.ERROR)


def warning(msg: str) -> None:
    """Convenvience warning logging function.

    Args:
        msg (str): Warning message
    """
    log(msg, logging.WARNING)


def info(msg: str) -> None:
    """Convenvience info logging function.

    Args:
        msg (str): Info message
    """
    log(msg, logging.INFO)


def debug(msg: str) -> None:
    """Convenvience debug logging function.

    Args:
        msg (str): Debug message
    """
    log(msg, logging.DEBUG)


def setup(output: "Path", *, verbose: bool = False) -> None:
    """Setup SuPAErnova logging.

    Args:
        output ("Path"): Directory to write supaernova.log to. This log file will contain debug message regardless of log level.

    Kwargs:
        verbose (bool): Increase log verbosity. Defaults to False
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logfile = output / "supaernova.log"
    level = logging.DEBUG if verbose else logging.INFO

    fmt_debug = "[%(levelname)8s] %(filename)10s | %(message)s"
    fmt_info = "%(message)s"
    fmt = fmt_debug if verbose else fmt_info

    level_styles = coloredlogs.parse_encoded_styles(
        "debug=8;info=green;warning=yellow;error=red,bold;critical=red,inverse",
    )

    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        coloredlogs.ColoredFormatter(fmt_debug, level_styles=level_styles),
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(
        coloredlogs.ColoredFormatter(fmt, level_styles=level_styles),
    )
    logger.addHandler(stream_handler)

    debug(f"Setup logging with{'' if verbose else 'out'} verbose logs")
    info(f"Logging to {logfile}")
