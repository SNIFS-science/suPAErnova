from typing import TYPE_CHECKING, cast
from pathlib import Path
import traceback

import toml
import click

from suPAErnova.steps import Data, TF_AutoEncoder
from suPAErnova.utils import logging as log

if TYPE_CHECKING:
    from suPAErnova.steps import Step
    from suPAErnova.utils.typing import CFG, INPUT
    from suPAErnova.config.requirements import RequirementReturn

# --- Constants ---
STEPS: dict[str, type["Step"]] = {"DATA": Data, "TF_AUTOENCODER": TF_AutoEncoder}


# --- Utilities ---
def missing_step(step: str) -> None:
    log.error(f"Unknown Step: {step}")


def normalise_config(cfg: "INPUT") -> "CFG":
    rtn: CFG = {}
    for k, v in cfg.items():
        val = v
        if isinstance(v, dict):
            val = normalise_config(cast("CFG", v))
        rtn[k.upper()] = val
    return rtn


@click.command()
@click.option("-v", "--verbose", is_flag=True, type=bool, default=False)
@click.option("-f", "--force", is_flag=True, type=bool, default=False)
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def cli(verbose: bool, force: bool, config: Path) -> "RequirementReturn[None]":
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

    steps = list(filter(lambda step: step != "GLOBAL", cfg.keys()))
    # Setup Steps
    for name in steps:
        cls = STEPS.get(name.upper())
        if cls is None:
            missing_step(name)
            continue
        try:
            log.info(f"Setting up {name.upper()}")
            step = cls(cfg)
            cfg["GLOBAL"]["RESULTS"][step.name] = step
        except Exception:
            result = traceback.format_exc()
            log.exception(result)
            return False, result
    # Run Steps
    for step in cfg["GLOBAL"]["RESULTS"].values():
        try:
            log.info(f"Running {step.name}")
            ok, result = step.setup()
            if not ok:
                return ok, result
            ok, result = step.run()
            if not ok:
                return False, result
            ok, result = step.result()
            if not ok:
                return False, result
            cfg = result
        except Exception:
            result = traceback.format_exc()
            log.exception(result)
            return False, result
    # Run any analysis
    for step in cfg["GLOBAL"]["RESULTS"].values():
        try:
            log.info(f"Analysing {step.name}")
            ok, result = step.analyse()
            if not ok:
                return False, result
        except Exception:
            result = traceback.format_exc()
            log.exception(result)
            return False, result
    return True, None


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
