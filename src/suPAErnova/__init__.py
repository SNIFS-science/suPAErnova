from typing import TYPE_CHECKING
from pathlib import Path
import traceback

import toml
import click

from suPAErnova.steps import Data, AutoEncoder
from suPAErnova.utils import logging as log

if TYPE_CHECKING:
    from suPAErnova.steps import Step
    from suPAErnova.utils.typing import CFG, INPUT

# --- Constants ---
STEPS: dict[str, type["Step"]] = {"DATA": Data, "AUTOENCODER": AutoEncoder}


# --- Utilities ---
def missing_step(step: str) -> None:
    log.error(f"Unknown Step: {step}")


def normalise_config(cfg: "INPUT") -> "CFG":
    rtn: CFG = {}
    for k, v in cfg.items():
        val = v
        if isinstance(v, dict):
            val = normalise_config(v)
        rtn[k.upper()] = val
    return rtn


@click.command()
@click.option("-v", "--verbose", is_flag=True, type=bool, default=False)
@click.option("-f", "--force", is_flag=True, type=bool, default=False)
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def cli(verbose: bool, force: bool, config: Path) -> bool:
    cfg: CFG = normalise_config(toml.load(config))

    # Setup global config
    cfg["GLOBAL"] = cfg.get("GLOBAL", {})

    # Set verbosity
    cfg["GLOBAL"]["VERBOSE"] = verbose

    # Force rerun
    cfg["GLOBAL"]["FORCE"] = force

    # Set base path
    basepath: Path | str = cfg["GLOBAL"].get("BASE", config.parent)
    base: Path = Path(basepath).resolve()
    cfg["GLOBAL"]["BASE"] = base

    # Set output path
    output = cfg["GLOBAL"].get("OUTPUT")
    outpath = Path(output) if isinstance(output, str) else Path("output")
    if not outpath.is_absolute():
        outpath = base / outpath
    outpath = outpath.resolve()
    if not outpath.exists():
        outpath.mkdir(parents=True)
    cfg["GLOBAL"]["OUTPUT"] = outpath

    # Store results
    cfg["GLOBAL"]["RESULTS"] = {}

    # Setup logging
    log.setup(verbose, outpath)
    cfg["GLOBAL"]["LOG"] = log

    # Run Steps
    steps = list(filter(lambda step: step != "GLOBAL", cfg.keys()))
    for step in steps:
        cls = STEPS.get(step.upper())
        if cls is None:
            missing_step(step)
            continue
        try:
            cfg = cls(cfg).run().result()
        except Exception:
            log.exception(traceback.format_exc())
            return False

    # Run any analysis
    for step in cfg["GLOBAL"]["RESULTS"].values():
        try:
            step.analyse()
        except Exception:
            log.exception(traceback.format_exc())
            return False
    return True


def main() -> None:
    cli()
