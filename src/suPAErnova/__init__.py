# --- External Imports ---
from typing import cast
import click
import toml
from pathlib import Path

# --- Internal Imports ---
from suPAErnova.utils import logging as log
from suPAErnova.data import Data
from suPAErnova.utils.typing import CFG, INPUT

# --- Constants ---
STEPS = {"DATA": Data}


def missing_step(step: str):
    log.error(f"Unknown Step: {step}")


@click.command()
@click.option("-v", "--verbose", is_flag=True, type=bool, default=False)
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def cli(verbose: bool, config: Path):
    input_cfg: INPUT = toml.load(config)
    cfg: CFG = input_cfg

    # Setup global config
    cfg["global"] = cast(CFG, input_cfg.get("global", {}))

    # Set verbosity
    cfg["global"]["verbose"] = verbose

    # Set base path
    basepath: Path | str = cfg["global"].get("base", config.parent)
    base: Path = Path(basepath).resolve()
    cfg["global"]["base"] = base

    # Set output path
    output = cfg["global"].get("output")
    if isinstance(output, str):
        outpath = Path(output)
    else:
        outpath = Path("output")
    if not outpath.is_absolute():
        outpath = base / outpath
    outpath = outpath.resolve()
    if not outpath.exists():
        outpath.mkdir(parents=True)
    cfg["global"]["output"] = outpath

    # Store results
    cfg["global"]["results"] = {}

    # Setup logging
    log.setup(verbose, outpath)
    cfg["global"]["log"] = log

    # Run Steps
    steps = list(filter(lambda step: step != "global", cfg.keys()))
    for step in steps:
        cls = STEPS.get(step.upper())
        if cls is None:
            missing_step(step)
            continue
        try:
            cfg = cls(cfg).run().result()
        except Exception as e:
            log.error(str(e))
            return False


def main() -> None:
    cli()
