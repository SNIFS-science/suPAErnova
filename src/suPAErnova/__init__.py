# Copyright 2025 Patrick Armstrong
"""Implements a probability autoencoder (PAE) to standardise SN Ia from spectra.

SuPAErnova provides tools to read in SN spectral data, train a PAE, and use the trained model to simulate and fit spectra. See [doi](10.3847/1538-4357/ac7c08), [arXiv](https://arxiv.org/abs/2207.07645) for more details.
"""

import sys
from typing import TYPE_CHECKING, cast
from pathlib import Path
import traceback
import contextlib

import toml
import click
from pydantic import ValidationError
from tqdm.contrib.logging import logging_redirect_tqdm

from suPAErnova.configs import SNPAEConfig
from suPAErnova.logging import setup_logging
from suPAErnova.configs.input import InputConfig
from suPAErnova.configs.paths import PathConfig
from suPAErnova.configs.steps import StepConfig
from suPAErnova.configs.config import GlobalConfig

if TYPE_CHECKING:
    from pydantic import JsonValue


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, type=bool, default=False)
@click.option("-f", "--force", is_flag=True, type=bool, default=False)
@click.option(
    "-b",
    "--base_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "-o",
    "--out_path",
    type=click.Path(exists=False, path_type=Path),
    default=None,
)
def cli(
    input_path: Path,
    *,  # Force keyword-only arguments
    verbose: bool = False,
    force: bool = False,
    base_path: Path | None = None,
    out_path: Path | None = None,
) -> None:
    input_config: dict[str, JsonValue] = toml.load(input_path)

    # Set base_path to input_path.parent if none provided
    base_path = PathConfig.resolve_path(
        base_path,
        default_path=input_path.parent,
        relative_path=Path.cwd(),
    )

    return main(
        input_config,
        verbose=verbose,
        force=force,
        base_path=base_path,
        out_path=out_path,
    )


def main(
    input_config: dict[str, "JsonValue"],
    *,  # Force keyword-only arguments
    verbose: bool = False,
    force: bool = False,
    base_path: Path | None = None,
    out_path: Path | None = None,
) -> None:
    log = setup_logging(__name__, verbose=verbose)
    log.info("Started SuPAErnova")

    try:
        # Setup context based on logging verbosity
        #   If verbose, redirect logging stdout through tqdm
        #   Otherwise use a nullcontext (which does nothing)
        cm = logging_redirect_tqdm() if verbose else contextlib.nullcontext()
        with cm:
            # Normalise input_config
            user_config = SNPAEConfig.normalise_input(input_config)

            # Setup global config
            user_config["config"] = GlobalConfig.from_config(
                cast("dict[str, JsonValue]", user_config.get("config", {})),
                verbose=verbose,
                force=force,
            )

            # Setup  paths config
            base_path = PathConfig.resolve_path(
                base_path,
                default_path=Path.cwd(),
                relative_path=Path.cwd(),
            )
            out_path = PathConfig.resolve_path(
                out_path,
                default_path=base_path / "output",
                relative_path=base_path,
                mkdir=True,
            )
            log_path = PathConfig.resolve_path(
                out_path / "logs",
                default_path=out_path / "logs",
                relative_path=base_path,
                mkdir=True,
            )
            # Re-setup base logger to include log file
            log = setup_logging(__name__, log_path=log_path.parent, verbose=verbose)
            user_config["paths"] = PathConfig.from_config(
                cast("dict[str, JsonValue]", user_config.get("paths", {})),
                base_path=base_path,
                out_path=out_path,
                log_path=log_path,
            )

            # Propagate global and paths to steps
            for step, step_config in StepConfig.steps.items():
                if step in user_config:
                    user_config[step] = step_config.from_config({
                        "config": user_config["config"],
                        "paths": user_config["paths"],
                        **user_config[step],
                    })

            config = InputConfig.from_config(user_config)
            config.run()
    except ValidationError as e:
        log.error(e)  # noqa: TRY400
        sys.exit(1)
    except Exception:
        log.exception(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    cli()
