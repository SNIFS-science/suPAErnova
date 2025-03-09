# Copyright 2025 Patrick Armstrong
"""Implements a probability autoencoder (PAE) to standardise SN Ia from spectra.

SuPAErnova provides tools to read in SN spectral data, train a PAE, and use the trained model to simulate and fit spectra. See [doi](10.3847/1538-4357/ac7c08), [arXiv](https://arxiv.org/abs/2207.07645) for more details.
"""

from typing import TYPE_CHECKING, cast
from pathlib import Path
import traceback

import toml
import click

from suPAErnova.steps import Data, TF_AutoEncoder
from suPAErnova.utils import suPAErnova_logging as log

if TYPE_CHECKING:
    from suPAErnova.steps import Step
    from suPAErnova.config.requirements import RequirementReturn
    from suPAErnova.utils.suPAErnova_types import CFG, INPUT


#
# === Constants ===
#


STEPS: dict[str, type["Step"]] = {"DATA": Data, "TF_AUTOENCODER": TF_AutoEncoder}


#
# === Utilities ===
#


def normalise_config(config: "INPUT") -> "CFG":
    """Prepare a config for use by forcing all keys to be capitalised.

    Args:
        config (INPUT): The config to normalise

    Returns:
        CFG: The normalised config
    """
    rtn: CFG = {}
    for k, v in config.items():
        val = v
        if isinstance(v, dict):
            val = normalise_config(cast("CFG", v))
        rtn[k.upper()] = val
    return rtn


def setup_steps(cfg: "CFG", steps: list[str]) -> "RequirementReturn[CFG]":
    """Run each step's setup() function. Post setup, each step will be stored in cfg["GLOBAL"]["RESULTS"][step.name].

    Args:
        cfg (CFG): The config used to control and store each step's setup
        steps (list[str]): The name of each step to setup. Available steps are listed in STEPS

    Returns:
        RequirementReturn[CFG]: The post setup config
    """
    for name in steps:
        cls = STEPS.get(name.upper())
        if cls is None:
            log.error(f"Unknown Step: {name}")
            continue
        try:
            log.info(f"Setting up {name.upper()}")
            step = cls(cfg)
            cfg["GLOBAL"]["RESULTS"][step.name] = step
        except Exception:
            result = traceback.format_exc()
            log.exception(result)
            return False, result
    return True, cfg


def run_steps(cfg: "CFG") -> "RequirementReturn[CFG]":
    """Run each step's run() function. The results of each run are stored in cfg["GLOBAL"]["RESULTS"][step.name].

    Args:
        cfg (CFG): The config used to control each step's run, and store the result

    Returns:
        RequirementReturn[CFG]: The post run config
    """
    # The only way success and result are unbound is if there are no results to run
    success = False
    result = "No steps available to run"

    steps = cfg["GLOBAL"]["RESULTS"].values()
    for step in steps:
        step = cast("Step", step)
        try:
            log.info(f"Running {step.name}")
            success, result = step.setup()
            if success:
                success, result = step.run()
            if success:
                success, result = step.result()
            if success:
                cfg = cast("CFG", result)
        except Exception:
            success = False
            result = traceback.format_exc()
            log.exception(result)

    result = cast("str", result) if not success else cast("CFG", result)
    return success, result


def run_analysis(cfg: "CFG") -> "RequirementReturn[None]":
    """Run each step's analyse() function.

    Args:
        cfg (CFG): The config used to control each step's analysis

    Returns:
        RequirementReturn[None]
    """
    # The only way success and result are unbound is if there are no results to run
    success = False
    result = "No steps available to run"

    steps = cfg["GLOBAL"]["RESULTS"].values()
    for step in steps:
        step = cast("Step", step)
        try:
            log.info(f"Analysing {step.name}")
            success, result = step.analyse()
        except Exception:
            success = False
            result = traceback.format_exc()
            log.exception(result)

    return success, result


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, type=bool, default=False)
@click.option("-f", "--force", is_flag=True, type=bool, default=False)
def cli(config: Path, *, verbose: bool, force: bool) -> "RequirementReturn[None]":
    """Command line interface for SuPAErnova.

    Args:
        config (Path): Path to the config.toml containing your desired configuration

    Kwargs:
        verbose (bool): Increase log verbosity. Defaults to False
        force (bool): Force every step to run, even if a previous result already exists

    Returns:
        RequirementReturn[None]
    """
    cfg = toml.load(config)
    return main(cfg, verbose=verbose, force=force, basepath=config.parent)


def main(
    config: "CFG",
    *,
    verbose: bool = False,
    force: bool = False,
    basepath: Path | None = None,
) -> "RequirementReturn[None]":
    """Main SuPAErnova entry point, executing each of the steps defined in config.

    Args:
        config (CFG): Path to the config.toml containing your desired configuration

    Kwargs:
        verbose (bool): Increase log verbosity. Defaults to False
        force (bool): Force every step to run, even if a previous result already exists. Defaults to False
        basepath (Path): The directory from which all paths are assumed to be relative. Defaults to Path.cwd()

    Returns:
        RequirementReturn[None]
    """
    cfg: CFG = normalise_config(config)

    # Setup global config
    cfg["GLOBAL"] = cfg.get("GLOBAL", {})

    # Set verbosity
    cfg["GLOBAL"]["VERBOSE"] = verbose

    # Force rerun
    cfg["GLOBAL"]["FORCE"] = force

    # Set base path to:
    # - cfg["GLOBAL"]["BASE"] or
    # - basepath or
    # - Path.cwd()
    base = Path(cfg["GLOBAL"].get("BASE", basepath or Path.cwd())).resolve()
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
    log.setup(outpath, verbose=verbose)
    cfg["GLOBAL"]["LOG"] = log

    # Determin what steps to run
    steps = list(filter(lambda step: step != "GLOBAL", cfg.keys()))

    # Setup Steps
    success, result = setup_steps(cfg, steps)
    if success:
        cfg = cast("CFG", result)

        # Run Steps
        success, result = run_steps(cfg)

    if success:
        cfg = cast("CFG", result)

        # Run analysis
        success, result = run_analysis(cfg)

    if not success:
        result = cast("str", result)
        log.error(result)
        return False, result
    return True, None


if __name__ == "__main__":
    cli()
