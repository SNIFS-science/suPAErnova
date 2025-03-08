from typing import TYPE_CHECKING, cast
from pathlib import Path

from astropy import cosmology as cosmo
import sncosmo

from suPAErnova.config.requirements import Requirement

if TYPE_CHECKING:
    from suPAErnova.config.requirements import REQ
    from suPAErnova.utils.suPAErnova_types import CFG

#
# === Requirements ===
#


# --- Data Directory ---
def valid_data_dir(datapath: str, cfg: "CFG", _: "CFG"):
    path = Path(datapath)
    if not path.is_absolute():
        path = cfg["BASE"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    return True, path


data_dir = Requirement[str, Path](
    "data_dir",
    "Path to directory containing data",
    transform=valid_data_dir,
)


# --- Meta File ---
def valid_meta_file(metapath: str, _: "CFG", opts: "CFG"):
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
    "Metadata CSV containing SN names and SALT fit parameters",
    transform=valid_meta_file,
)


def valid_idr_file(idrpath: str, _: "CFG", opts: "CFG"):
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
    "A seperate file containing additional SALT fit parameters (dphase)",
    transform=valid_idr_file,
)


def valid_mask_file(maskpath: str, _: "CFG", opts: "CFG"):
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
    "A mask of bad spectra / wavelength range",
    transform=valid_mask_file,
)

#
# === Optional ===
#

# --- SALT2 ---
salt_cosmo = Requirement[str, cosmo.FlatLambdaCDM](
    "cosmo",
    "The cosmology to use when running SALT2 models",
    default="WMAP7",
    choice=cosmo.realizations.available,
    transform=lambda name, _1, _2: (True, getattr(cosmo, name)),
)


def gen_salt_model(salt: str, _1: "CFG", _2: "CFG"):
    if Path(salt).exists():
        if "salt2" in salt:
            source = sncosmo.SALT2Source(salt)
        elif "salt3" in salt:
            source = sncosmo.SALT3Source(salt)
        else:
            return (
                False,
                f"SALT model {salt} does not seem to be a salt2 or salt3 model",
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
                f"SALT source {salt} does not seem to be a salt2 or salt3 source",
            )
    return True, source


salt_model = Requirement[str, sncosmo.SALT2Source | sncosmo.SALT3Source](
    "salt",
    "The absolute Path to an existing SALT model, or the name of an existing SNCosmo source",
    default="salt3",
    transform=gen_salt_model,
)

min_phase = Requirement[int | float, float](
    name="min_phase",
    description="Cut spectral data earlier than this phase",
    default=-10.0,
    transform=lambda phase, _1, _2: (True, float(phase)),
)

max_phase = Requirement[int | float, float](
    name="max_phase",
    description="Cut spectral data later than this phase",
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
    description="Fraction of total data to be used for training",
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
    description="Random seed used for splitting",
    default=12345,
)


required: list["REQ"] = [data_dir, meta_file, idr_file, mask_file]
optional: list["REQ"] = [
    salt_cosmo,
    salt_model,
    min_phase,
    max_phase,
    train_frac,
    train_seed,
]
