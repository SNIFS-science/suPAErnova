from pathlib import Path
from typing import final, override
import numpy as np
import pandas as pd
from tqdm import tqdm

from suPAErnova.steps import Requirement, Step
from suPAErnova.utils.typing import CFG


# === Requirements ===
# --- Data Directory ---
def valid_data_dir(path: Path, cfg: CFG, _: CFG):
    if not path.is_absolute():
        path = cfg["base"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    return True, path


data_dir = Requirement[str, Path](
    "data_dir",
    "Path to directory containing data",
    transform=lambda path, _1, _2: Path(path),
    valid_transform=valid_data_dir,
)


# --- Meta File ---
def valid_meta_file(path: Path, _: CFG, opts: CFG):
    if not path.is_absolute():
        path = opts["data_dir"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    if path.suffix != ".csv":
        return False, f"{path} is not a .csv file"
    return True, path


meta_file = Requirement[str, Path](
    "meta",
    "Metadata CSV containing SN names and SALT fit parameters",
    transform=lambda path, _1, _2: Path(path),
    valid_transform=valid_meta_file,
)


def valid_idr_file(path: Path, _: CFG, opts: CFG):
    if not path.is_absolute():
        path = opts["data_dir"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    if path.suffix != ".txt":
        return False, f"{path} is not a .txt file"
    return True, path


idr_file = Requirement[str, Path](
    "idr",
    "A seperate file containing additional SALT fit parameters (dphase)",
    transform=lambda path, _1, _2: Path(path),
    valid_transform=valid_idr_file,
)


def valid_mask_file(path: Path, _: CFG, opts: CFG):
    if not path.is_absolute():
        path = opts["data_dir"] / path
    path = path.resolve()
    if not path.exists():
        return False, f"{path} does not exist"
    if path.suffix != ".txt":
        return False, f"{path} is not a .txt file"
    return True, path


mask_file = Requirement[str, Path](
    "mask",
    "A mask of bad spectra / wavelength range",
    transform=lambda path, _1, _2: Path(path),
    valid_transform=valid_mask_file,
)


@final
class Data(Step):
    required = [data_dir, meta_file, idr_file, mask_file]

    def __init__(self, cfg: CFG):
        super().__init__(cfg)

        # Metadata
        self.metapath: Path = self.opts["meta"]
        self.idrpath: Path = self.opts["idr"]
        self.maskpath: Path = self.opts["mask"]

        # The following parameters get filled in self._run()

        # Generated Data
        self.data: CFG | None = None

    def get_sn_salt_data(self):
        self.log.debug("Gathering metadata")
        # Header:
        #   id:           str   = Spectra Id
        #   sn:           str   = SN Name
        #   phase:        float = Spectral phase relative to peak mag in days
        #   z:            float = Redshift of SN
        #   MB:           float = Redshift-dependant absolute magnitude of a ``standard'' SN Ia
        #   x0:           float = SALT $x_{0}$ parameter, with the SN apparent magnitude $m_{b}=\log_{10}(x0)$
        #   x1:           float = SALT $x_{1}$ stretch parameter
        #   c:            float = SALT $\mathcal{C}$ colour parameter
        #   path:         str   = Path to spectra, relative to metapath
        #   hubble_resid: float = Hubble Residual
        meta = pd.read_csv(self.metapath, header=0)
        # Update paths relative to metapath
        meta["path"] = meta["path"].map(lambda path: self.metapath.parent / path)

        self.log.debug("Gathering additional metadata")
        # Header:
        #   sn:     str   = SN Name -> salt.sn
        #   mjd:    float = Time of spectra obsevation in MJD
        #   dphase: float = Phase offset
        idr = pd.read_csv(self.idrpath, sep="\\s+", names=["sn", "mjd", "dphase"])
        # Merge by sn name
        meta = meta.merge(idr, on="sn", how="left")

        self.log.debug("Calculating SN Mask")
        mask = pd.read_csv(
            self.maskpath,
            sep="\\s+",
            names=["sn", "id", "flag", "wavelength_min", "wavelength_max"],
        )
        mask["id"] = mask["sn"] + "_" + mask["id"]
        meta = meta.merge(mask, on=["sn", "id"], how="left")

        self.log.debug("Merging per-SN SALT metadata")

        sne_cols = ["sn", "MB", "x0", "x1", "c", "hubble_resid", "mjd", "dphase"]
        sne = meta[sne_cols].drop_duplicates()

        spec_cols = ["id", "phase", "path", "flag", "wavelength_min", "wavelength_max"]
        spec = meta[spec_cols]

    def gen_data(self):
        data = {}

    @override
    def _run(self):
        self.get_sn_salt_data()
        self.gen_data()
        return True, None

    @override
    def _result(self):
        if self.data is None:
            return False, "Data has not been generated"
        self.results["datapath"] = self.outpath / "data.npy"
        self.results["data"] = self.data
        self.log.debug(f"Writing generated data to {self.results['data']}")
        np.save(self.results["datapath"], self.data)
        return True, None
