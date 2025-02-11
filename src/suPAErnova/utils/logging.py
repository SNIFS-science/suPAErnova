from typing import TYPE_CHECKING
import logging

import coloredlogs

if TYPE_CHECKING:
    from pathlib import Path


def log(msg: str, level: int) -> None:
    logger = logging.getLogger()
    logger.log(level, msg)


def exception(msg: str) -> None:
    logger = logging.getLogger()
    logger.error(msg)


def error(msg: str) -> None:
    log(msg, logging.ERROR)


def warning(msg: str) -> None:
    log(msg, logging.WARNING)


def info(msg: str) -> None:
    log(msg, logging.INFO)


def debug(msg: str) -> None:
    log(msg, logging.DEBUG)


def setup(verbose: bool, output: "Path") -> None:
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
