# Copyright 2025 Patrick Armstrong
"""Data step configuration."""

from typing import TYPE_CHECKING, cast
from pathlib import Path

from astropy import cosmology as cosmo
import sncosmo

from suPAErnova.config.requirements import Requirement

if TYPE_CHECKING:
    from suPAErnova.config.requirements import REQ, RequirementReturn
    from suPAErnova.utils.suPAErnova_types import CFG


#
# === Required ===
#


def valid_data_dir(datapath: str, cfg: "CFG", _: "CFG") -> "RequirementReturn[Path]":
    """DATA_DIR: Set relative to cfg["BASE"] if not absolute, and ensure it exists.

    Args:
        datapath (str): User provide path to data
        cfg (CFG): Global configuration

    Returns:
        RequirementReturn[Path]
    """
    path = Path(datapath)
    if not path.is_absolute():
        path = cfg["BASE"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    return True, path


data_dir = Requirement[str, Path](
    "data_dir",
    "Path to directory containing data.\n    Can be absolute or relative to the base path.",
    transform=valid_data_dir,
)


def valid_meta_file(metapath: str, _: "CFG", opts: "CFG") -> "RequirementReturn[Path]":
    """META: Set relative to opts["DATA_DIR"] if not absolute, ensure it exists, and that it is a csv file.

    Args:
        metapath (str): User provide path to meta.csv
        opts (CFG): Data step specific options

    Returns:
        RequirementReturn[Path]
    """
    path = Path(metapath)
    if not path.is_absolute():
        path = opts["DATA_DIR"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    if path.suffix != ".csv":
        return False, f"{path} is not a .csv file"
    return True, path


meta_file = Requirement[str, Path](
    "meta",
    "Metadata CSV containing SN names and SALT fit parameters.\n    Can be absolute or relative to the data path.",
    transform=valid_meta_file,
)


def valid_idr_file(idrpath: str, _: "CFG", opts: "CFG") -> "RequirementReturn[Path]":
    """IDR: Set relative to opts["DATA_DIR"] if not absolute, ensure it exists, and that it is a txt file.

    Args:
        idrpath (str): User provide path to IDR_eTmax.txt
        opts (CFG): Data step specific options

    Returns:
        RequirementReturn[Path]
    """
    path = Path(idrpath)
    if not path.is_absolute():
        path = opts["DATA_DIR"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    if path.suffix != ".txt":
        return False, f"{path} is not a .txt file"
    return True, path


idr_file = Requirement[str, Path](
    "idr",
    "TXT file containing additional SALT fit parameters.\n    Can be absolute or relative to the data path.",
    transform=valid_idr_file,
)


def valid_mask_file(maskpath: str, _: "CFG", opts: "CFG") -> "RequirementReturn[Path]":
    """MASK: Set relative to opts["DATA_DIR"] if not absolute, ensure it exists, and that it is a txt file.

    Args:
        maskpath (str): User provide path to mask_info_wmin_wmax.txt
        opts (CFG): Data step specific options

    Returns:
        RequirementReturn[Path]
    """
    path = Path(maskpath)
    if not path.is_absolute():
        path = opts["DATA_DIR"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    if path.suffix != ".txt":
        return False, f"{path} is not a .txt file"
    return True, path


mask_file = Requirement[str, Path](
    "mask",
    "TXT file containing a mask of bad spectra / wavelength ranges.\n    Can be absolute or relative to the data path.",
    transform=valid_mask_file,
)

#
# === Optional ===
#

# --- SALT ---
cosmological_model = Requirement[str, cosmo.FlatLambdaCDM](
    "cosmological_model",
    "Which assumed cosmology to use when running SALT models.\n    Available cosmological can be found [here](https://docs.astropy.org/en/stable/cosmology/realizations.html)\n    Defaults to WMAP7",
    default="WMAP7",
    choice=cosmo.realizations.available,
    transform=lambda name, _1, _2: (True, getattr(cosmo, name)),
)


def gen_salt_model(
    salt: str,
    _1: "CFG",
    _2: "CFG",
) -> "RequirementReturn[sncosmo.SALT2Source | sncosmo.SALT3Source]":
    """SALT: Check whether `salt` is an absolute path to an existing SALT model, or an existing SNCosmo SALT source.

    Args:
        salt (str): User provided SALT model specification

    Returns:
        RequirementReturn[SALT2Source | SALT3Source]
    """
    if Path(salt).exists():
        if "salt2" in salt:
            source = sncosmo.SALT2Source(salt)
        elif "salt3" in salt:
            source = sncosmo.SALT3Source(salt)
        else:
            return (
                False,
                f"SALT model {salt} does not seem to be a SALT2 or SALT3 model",
            )
    else:
        source = sncosmo.get_source(salt)
        if "salt2" in salt:
            source = cast("sncosmo.SALT2Source", source)
        elif "salt3" in salt:
            source = cast("sncosmo.SALT3Source", source)
        else:
            return (
                False,
                f"SALT source {salt} does not seem to be a SALT2 or SALT3 source",
            )
    return True, source


salt_model = Requirement[str, sncosmo.SALT2Source | sncosmo.SALT3Source](
    "salt_model",
    "The absolute path to an existing SALT2/3 model, or the name of an existing SNCosmo SALT2/3 model.\n    Defaults to salt3",
    default="salt3",
    transform=gen_salt_model,
)

min_phase = Requirement[int | float, float](
    name="min_phase",
    description="Minimum phase for spectral data, relative to peak. Spectral data earlier than this phase will be cut.\n    Defaults to -10.0",
    default=-10.0,
    transform=lambda phase, _1, _2: (True, float(phase)),
)

max_phase = Requirement[int | float, float](
    name="max_phase",
    description="Maximum phase for spectral data, relative to peak. Spectral data later than this phase will be cut.\n    Defaults to 40.0",
    default=40.0,
    transform=lambda phase, _, opts: (True, float(phase))
    if phase > opts["MIN_PHASE"]
    else (
        False,
        f"max_phase={phase} must be greater than min_phase={opts['MIN_PHASE']}",
    ),
)

train_frac = Requirement[int | float, float](
    name="train_frac",
    description="The fraction of data to be used for training, with the rest of the data going to testing and validation.\n    Defaults to 0.75",
    default=0.75,
    transform=lambda frac, _1, _2: (
        (True, frac)
        if 0 < frac < 1
        else (
            False,
            f"Invalid train_frace: {frac} must be between 0 and 1",
        )
    ),
)

train_seed = Requirement[int, int](
    name="seed",
    description="The seed used throughout data preperation, in particular for randomly splitting the data into training, testing, and validation bins.\n    Defaults to 12345",
    default=12345,
)


required: list["REQ"] = [data_dir, meta_file, idr_file, mask_file]
optional: list["REQ"] = [
    cosmological_model,
    salt_model,
    min_phase,
    max_phase,
    train_frac,
    train_seed,
]
