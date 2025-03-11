from typing import TYPE_CHECKING, cast, final, override

import numpy as np
import pandas as pd

from suPAErnova.steps import Step, callback
from suPAErnova.config.data import optional, required

if TYPE_CHECKING:
    from typing import TypeVar
    from pathlib import Path
    from collections.abc import Iterable, Sequence

    from astropy import cosmology as cosmo
    import sncosmo
    import numpy.typing as npt

    from suPAErnova.config.requirements import RequirementReturn
    from suPAErnova.utils.suPAErnova_types import CFG

    T = TypeVar("T", bound=np.generic)


@final
class DATAStep(Step):
    required = required
    optional = optional

    def __init__(self, cfg: "CFG") -> None:
        super().__init__(cfg)

        # Default
        self.datapath = self.outpath / "data.npz"
        self.snepath = self.outpath / "sne.pkl"
        self.trainpath = self.outpath / "train"
        if not self.trainpath.exists():
            self.trainpath.mkdir(parents=True)
        self.testpath = self.outpath / "test"
        if not self.testpath.exists():
            self.testpath.mkdir(parents=True)

        # Load from config file
        # Required
        self.metapath: Path
        self.idrpath: Path
        self.maskpath: Path

        # Optional
        # SALT Setup
        self.cosmo: cosmo.FlatLambdaCDM
        self.snmodel: sncosmo.SALT2Source | sncosmo.SALT3Source

        # Quality Cuts
        self.min_phase: int
        self.max_phase: int

        # Train and Test split
        self.train_frac: float
        self.test_frac: float
        self.nkfold: int
        self.seed: int
        self.rng: np.random.Generator

        # Generate in self._run()
        # Lengths
        self.n_sn: int
        self.nspectra_per_sn: npt.NDArray[np.int32]
        self.wavelength: pd.Series
        self.n_timemax: int
        self.n_timemin: int
        self.n_wavelength: int

        # Data Products
        self.sne: pd.DataFrame
        self.data: CFG
        self.train_data: list[CFG]
        self.test_data: list[CFG]

    @override
    def __str__(self) -> str:
        rtn = super().__str__()
        if self.is_setup:
            lines = [
                f"    Data Path: {self.datapath}",
                f"    SNe Path: {self.snepath}",
                f"    Train Path: {self.trainpath}",
                f"    Test Path: {self.testpath}",
                f"    Meta Path: {self.metapath}",
                f"    IDR Path: {self.idrpath}",
                f"    Mask Path: {self.maskpath}",
                f"    Cosmological Model: {self.cosmo}",
                f"    SALT Model: {self.snmodel}",
                f"    Min Phase: {self.min_phase}",
                f"    Max Phase: {self.max_phase}",
                f"    Training Fraction: {self.train_frac}",
                f"    Testing Fraction: {self.test_frac}",
                f"    Number of KFolds: {self.nkfold}",
                f"    Seed: {self.seed}",
            ]
            rtn += "\n\n" + "\n".join(lines)
        if self.has_run:
            lines = [
                f"    Number of SNe: {self.n_sn}",
                f"    Number of wavelength elements: {self.n_wavelength}",
                f"    Number of Spectra per SNe: {self.nspectra_per_sn}",
                f"    Maximum number of Spectra per SNe: {self.n_timemax}",
                f"    Minimum number of Spectra per SNe{self.n_timemin}",
                f"    Data: {self.data}",
                f"    Training Data: {self.train_data}",
                f"    Testing Data: {self.test_data}",
            ]
            rtn += "\n\n" + "\n".join(lines)
        return rtn

    @override
    def _setup(self):
        # Load from config file
        # Required
        self.metapath = cast("Path", self.opts["META"])
        self.idrpath = cast("Path", self.opts["IDR"])
        self.maskpath = cast("Path", self.opts["MASK"])

        # Optional
        # SALT Setup
        self.cosmo = cast("cosmo.FlatLambdaCDM", self.opts["COSMOLOGICAL_MODEL"])
        self.snmodel = cast(
            "sncosmo.SALT2Source | sncosmo.SALT3Source",
            self.opts["SALT_MODEL"],
        )
        self.log.debug(f"SALT model parameters are: {self.snmodel.param_names}")

        # Quality Cuts
        self.min_phase = cast("int", self.opts["MIN_PHASE"])
        self.max_phase = cast("int", self.opts["MAX_PHASE"])

        # Train and Test split
        self.train_frac = cast("float", self.opts["TRAIN_FRAC"])
        self.test_frac = 1 - self.train_frac
        self.nkfold = int(1.0 / self.test_frac)
        self.seed = self.opts["SEED"]
        self.rng = np.random.default_rng(self.seed)

        self.train_data = []
        self.test_data = []

        return (True, None)

    @callback
    def get_dims(self) -> None:
        # --- Get Dimensions ---
        # Number of SNe
        self.n_sn = n_sn = len(self.sne)
        self.log.debug(f"Number of SNe: {n_sn}")

        # Maximum number of observations for any given SN
        self.nspectra_per_sn = np.array(
            [len(spectra) for spectra in self.sne["spectra"]],
        )
        self.n_timemax = self.nspectra_per_sn.max()
        self.n_timemin = self.nspectra_per_sn.min()
        self.log.debug(
            f"Maximum number of observations for any given SN: {self.n_timemax}",
        )

        # Wavelength grid
        # Since all spectra share the same wavelength grid
        # Just get the wavelength grid of the first spectrum
        self.wavelength = cast("pd.Series", self.sne["spectra"][0]["data"][0]["wave"])
        self.n_wavelength = len(self.wavelength)
        self.log.debug(f"Length of wavelength grid: {self.n_wavelength}")

    #
    # === Required Functions ===
    #

    @callback
    def load_data(self) -> None:
        self.log.debug("Loading SN SALT data")
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
        sn_data = pd.read_csv(self.metapath, header=0)
        # Update paths relative to metapath
        sn_data["path"] = sn_data["path"].apply(
            lambda path: self.metapath.parent / path,
        )

        self.log.debug("Loading SN DPhase data")
        # Header:
        #   sn:     str   = SN Name -> salt.sn
        #   mjd:    float = Time of spectra obsevation in MJD
        #   dphase: float = Phase offset
        dphase_data = pd.read_csv(
            self.idrpath,
            sep="\\s+",
            names=["sn", "mjd", "dphase"],
        )
        # Merge by sn name
        sn_data = sn_data.merge(dphase_data, on="sn", how="left")

        self.log.debug("Loading Spectra Masks")
        # Header:
        #   sn:             str   = SN Name
        #   id:             str   = Spectra Id
        #   flag:           int   = Quality of spectra
        #   wavelength_min: float = Min wavelength of spectra
        #   wavelength_max: float = Max wavelength of spectra
        spectra_mask = pd.read_csv(
            self.maskpath,
            sep="\\s+",
            names=["sn", "id", "flag", "wavelength_min", "wavelength_max"],
        )
        spectra_mask["id"] = spectra_mask["sn"] + "_" + spectra_mask["id"]
        sn_data = sn_data.merge(spectra_mask, on=["sn", "id"], how="left")

        self.log.debug("Merging SN data")
        # Split data into two dataframes
        # A SN dataframe which contains one row per SN, and the following columns
        sne_cols = ["sn", "MB", "x0", "x1", "c", "z", "hubble_resid", "mjd", "dphase"]
        sne = cast(
            "pd.DataFrame",
            sn_data[sne_cols].drop_duplicates().reset_index(drop=True),
        )

        # A Spectra dtaaframe which contains one row per spectra, and the following columns
        # Note that we keep the sn column so that we can match each spectra with their SN
        spec_cols = [
            "sn",
            "id",
            "phase",
            "path",
            "flag",
            "wavelength_min",
            "wavelength_max",
        ]
        spectra = cast("pd.DataFrame", sn_data[spec_cols].reset_index(drop=True))

        self.log.debug(
            f"Cutting spectra with phases outside the range {self.min_phase} <= phase <= {self.max_phase}",
        )
        self.log.debug(f"Number of spectra before phase-cut: {len(spectra)}")
        spectra = cast(
            "pd.DataFrame",
            spectra[spectra["phase"].between(self.min_phase, self.max_phase)],
        )
        self.log.debug(f"Number of spectra after phase-cut: {len(spectra)}")

        self.log.debug("Loading Spectra data")
        spectra["data"] = [
            # Header:
            #   wave:  float = wavelength in AA
            #   flux:  float = flux
            #   sigma: float = flux error
            pd.read_csv(spec["path"])
            for _, spec in self.tqdm(
                spectra.iterrows(),
                desc="Loading Spectra",
                total=len(spectra),
            )
        ]

        self.log.debug("Linking Spectra with SNe")
        sne["spectra"] = cast("pd.Series", sne["sn"]).apply(
            lambda sn_name: spectra[spectra["sn"] == sn_name].reset_index(drop=True),
        )

        # Final structure is 1 row per SN with columns:
        #   sn:           str       = SN Name
        #   MB:           float     = Redshift-dependant absolute magnitude of a ``standard'' SN Ia
        #   x0:           float     = SALT $x_{0}$ parameter, with the SN apparent magnitude $m_{b}=\log_{10}(x0)$
        #   x1:           float     = SALT $x_{1}$ stretch parameter
        #   c:            float     = SALT $\mathcal{C}$ colour parameter
        #   z:            float     = Redshift of SN
        #   hubble_resid: float     = Hubble Residual
        #   dphase:       float     = Phase offset
        #   spectra:      DataFrame = SN Spectra with columns:
        #
        #       sn:             str       = SN Name
        #       id:             str       = Spectra Id
        #       phase:          float     = Spectral phase relative to peak mag in days
        #       path:           str       = Path to spectra, relative to metapath
        #       flag:           int       = Quality of spectra
        #       wavelength_min: float     = Min wavelength of spectra
        #       wavelength_max: float     = Max wavelength of spectra
        #       data:           DataFrame = Spectral data with columns:
        #
        #           wave:  Series[float]  = wavelength in AA
        #           flux:  Series[float]  = flux
        #           sigma: Series[float]  = flux error

        self.sne = sne

    @callback
    def calculate_salt_flux(self) -> None:
        def get_salt_flux(
            wavelength: "float | Iterable[float]",
            tobs: float = 0.0,
            z: float = 0.0,
            x0: float = 1.0,
            x1: float = 0.0,
            c: float = 0.0,
            zref: float = 0.05,
        ):
            self.snmodel.set(x0=x0, x1=x1, c=c)
            return (
                self.snmodel.flux(phase=tobs, wave=wavelength)
                * (
                    (
                        self.cosmo.luminosity_distance(z)
                        / self.cosmo.luminosity_distance(zref)
                    )
                    ** 2
                )
                * ((1 + z) / (1 + zref))
                * 1e15
            )

        for _, sn in self.sne.iterrows():
            for _, spectra in cast("pd.DataFrame", sn["spectra"]).iterrows():
                spectra["data"]["salt_flux"] = get_salt_flux(
                    cast("pd.Series", spectra["data"]["wave"]).to_numpy(),
                    tobs=cast("float", spectra["phase"]),
                    z=cast("float", sn["z"]),
                    x0=cast("float", sn["x0"]),
                    x1=cast("float", sn["x1"]),
                    c=cast("float", sn["c"]),
                )

    @callback
    def transform_data(self) -> None:
        # Each element of data is a 3D Array of shape (n_sn x n_timemax x n_data) where:
        #   n_sn = Number of SNe
        #   n_timemax = Maximum number of observations for any given SN (padded if needed)
        #   n_data = Length of datatype

        # Set dimensional parameters
        self.get_dims()

        # Allows for filling an array with padding
        time_axis = self.nspectra_per_sn.copy()
        time_axis.fill(self.n_timemax)

        # --- Get Parameters ---
        self.data = {}

        # Given an array of shape n_sn by N <= n_timemax
        # Create an array of shape n_sn by n_timemax, padding if needed
        def pad(
            arr: "Iterable[Sequence[T | npt.NDArray[T]]]",
            padding: "T | npt.NDArray[T]",
        ) -> "npt.NDArray[T]":
            if isinstance(padding, np.ndarray):
                padded_arr = np.full(
                    (self.n_sn, self.n_timemax, *padding.shape),
                    padding,
                )
            else:
                padded_arr = np.full((self.n_sn, self.n_timemax), padding)
            for i, row in enumerate(arr):
                row_length = len(row)
                padded_arr[i, :row_length] = row
            return padded_arr

        # Given a list of value-per-row of length n_sn
        # Fill each row with n_timemax repeats of that row's value
        def fill_rows(values: "npt.NDArray[T]") -> "npt.NDArray[T]":
            return np.repeat(values, time_axis).reshape((self.n_sn, self.n_timemax))

        # Index of each SNe
        self.data["ind"] = fill_rows(np.array(range(self.n_sn)))

        # Number of spectra per SNe
        self.data["nspectra"] = fill_rows(self.nspectra_per_sn)

        # Get SNe parameters
        sne_params = {
            "sn_name": "sn",
            "dphase": "dphase",
            "redshift": "z",
            "x0": "x0",
            "x1": "x1",
            "c": "c",
            "MB": "MB",
            "hubble_residual": "hubble_resid",
        }

        for data_key, sne_key in sne_params.items():
            self.data[data_key] = fill_rows(
                self.sne[sne_key].to_numpy(),
            )

        self.data["luminosity_distance"] = self.cosmo.luminosity_distance(
            self.data["redshift"],
        ).value

        # Get Parameters from spectra
        spectra_params = {
            "spectra_id": ("id", ""),
            "phase": ("phase", -100.0),
            "wl_mask_min": ("wavelength_min", -1.0),
            "wl_mask_max": ("wavelength_max", -1.0),
        }

        for data_key, (spectra_key, padding) in spectra_params.items():
            self.data[data_key] = pad(
                [spectra[spectra_key].to_numpy() for spectra in self.sne["spectra"]],
                padding,
            )

        # Get spectral data parameters
        spectral_data_params = {
            "flux": ("flux", np.zeros(self.n_wavelength)),
            "sigma": ("sigma", np.zeros(self.n_wavelength)),
            "salt_flux": ("salt_flux", np.zeros(self.n_wavelength)),
        }

        for data_key, (spectral_data_key, padding) in spectral_data_params.items():
            self.data[data_key] = pad(
                [
                    [
                        spectral_data[spectral_data_key].to_numpy()
                        for spectral_data in spectra["data"]
                    ]
                    for spectra in self.sne["spectra"]
                ],
                padding,
            )

        self.data["wavelength"] = pad(
            [
                [self.wavelength.to_numpy() for _ in spectra["data"]]
                for spectra in self.sne["spectra"]
            ],
            np.zeros(self.n_wavelength),
        )

        # Finally, ensure everything has the right number of axes
        for k, v in self.data.items():
            if len(v.shape) == 2:
                self.data[k] = v[..., np.newaxis]

    @callback
    def finalise_data(self) -> None:
        # Rescale phase to time such that:
        #   time = 0 -> phase = min_phase
        #   time = 1 -> phase = max_phase
        time_mask = self.data["phase"] > -100
        self.data["time"] = self.data["phase"].copy()
        self.data["time"][time_mask] = (
            self.data["time"][time_mask] - self.min_phase
        ) / (self.max_phase - self.min_phase)

        # Remove negative flux from unmasked fluxes
        self.data["flux"][self.data["mask"]] = np.clip(
            self.data["flux"][self.data["mask"]],
            0,
            np.inf,
        )

        # Scale observed uncertainty to account for fitting degrees of freedom, and an error floor
        self.data["sigma"] = 1.4 * self.data["sigma"] + 4e-10

    @callback
    def calculate_wavelength_mask(self) -> None:
        # Create a mask of wavelength outside of the wavelength limits
        self.data["mask"] = (self.data["wl_mask_min"] <= self.data["wavelength"]) & (
            self.data["wavelength"] <= self.data["wl_mask_max"]
        )

    @callback
    def calculate_laser_line_mask(self) -> None:
        # Mask any huge laser lines, Na D (5674 - 5692A)
        # TODO: Make these options
        # these are large jumps in flux, localized over a few wavelength bins
        wavelength_bin_start = 5000.0
        wavelength_bin_end = 8000.0
        laser_width = 2  # in units of wavelength bins
        laser_height = 0.4  # fractional increase in amplitude over neighbours to be considered laser

        wave_mask = (wavelength_bin_start <= self.data["wavelength"]) & (
            self.data["wavelength"] <= wavelength_bin_end
        )
        flux = np.zeros(self.data["flux"].shape)
        flux[wave_mask] = self.data["flux"][wave_mask]

        wave_mask_min = (
            wavelength_bin_start - laser_width <= self.data["wavelength"]
        ) & (self.data["wavelength"] <= wavelength_bin_end - laser_width)
        min_flux = np.zeros(self.data["flux"].shape)
        min_flux[wave_mask] = self.data["flux"][wave_mask_min]

        wave_mask_max = (
            wavelength_bin_start + laser_width <= self.data["wavelength"]
        ) & (self.data["wavelength"] <= wavelength_bin_end + laser_width)
        max_flux = np.zeros(self.data["flux"].shape)
        max_flux[wave_mask] = self.data["flux"][wave_mask_max]

        smooth_flux = 0.5 * (min_flux + max_flux)

        laser_mask = (flux - smooth_flux) > laser_height
        self.data["mask"] &= ~laser_mask

    @callback
    def split_train_test(self) -> None:
        # Train test split
        ind_split = int(self.n_sn * self.train_frac)

        # Select train_frac for training, the rest for testing
        inds = np.arange(0, self.n_sn)
        self.rng.shuffle(inds)

        # Split into k cross validation sets
        for kfold in range(self.nkfold):
            inds_k = np.roll(inds, kfold * inds.shape[0] // self.nkfold)

            inds_train = inds_k[:ind_split]
            inds_test = inds_k[ind_split:]

            self.train_data.append({
                key: val[inds_train, :, :] if val.ndim == 3 else val[inds_train, :]
                for key, val in self.data.items()
            })
            self.test_data.append({
                key: val[inds_test, :, :] if val.ndim == 3 else val[inds_test, :]
                for key, val in self.data.items()
            })
        n_train_sn = self.train_data[0]["flux"].shape[0]
        n_test_sn = self.test_data[0]["flux"].shape[0]
        self.log.debug(
            f"n_train_sn: {n_train_sn} ({100 * n_train_sn / self.n_sn}%) + n_test_sn: {n_test_sn} ({100 * n_test_sn / self.n_sn}%) = {n_train_sn + n_test_sn} ({100 * (n_train_sn + n_test_sn) / self.n_sn}%)",
        )

    @override
    def _is_completed(self) -> bool:
        return (
            self.datapath.exists()
            and self.snepath.exists()
            and self.trainpath.exists()
            and len(list(self.trainpath.iterdir())) > 0
            and self.testpath.exists()
            and len(list(self.trainpath.iterdir())) > 0
        )

    @override
    def _load(self):
        # Load SNe DataFrames
        self.sne = cast("pd.DataFrame", pd.read_pickle(self.snepath))
        # Calculate data dimensions
        self.get_dims()
        # Load data from files
        # Open the file, read each key into a dictionary, then close the file
        self.data = {}
        with np.load(self.datapath, allow_pickle=True) as io:
            for k, v in io.items():
                self.data[k] = v
        # Load in training and testing data
        self.train_data = []
        for train_data in self.trainpath.iterdir():
            if train_data.is_file():
                with np.load(train_data, allow_pickle=True) as io:
                    data = {}
                    for k, v in io.items():
                        data[k] = v
                    self.train_data.append(data)
        self.test_data = []
        for test_data in self.testpath.iterdir():
            if test_data.is_file():
                with np.load(test_data, allow_pickle=True) as io:
                    data = {}
                    for k, v in io.items():
                        data[k] = v
                    self.test_data.append(data)
        return True, None

    @override
    def _run(self):
        # Provides self.sne
        self.load_data()
        # Generate SALT fluxes
        self.calculate_salt_flux()
        # Provides self.data
        self.transform_data()
        # Provides self.data["mask"]
        self.calculate_wavelength_mask()
        # Provides self.data["mask"]
        self.calculate_laser_line_mask()
        # Provides self.data["time"], clips self.data["flux"] and scales self.data["sigma"]
        self.finalise_data()
        # Provides self.train and self.test
        self.split_train_test()
        return True, None

    @override
    def _result(self) -> "RequirementReturn[None]":
        if self.force or not self._is_completed():
            # Save SNe
            self.log.debug(f"Writing loaded SNe data to {self.snepath}")
            self.sne.to_pickle(self.snepath)

            # Save data
            self.log.debug(f"Writing generated data to {self.datapath}")
            np.savez_compressed(self.datapath, **self.data)

            # Save training data
            self.log.debug(f"Writing training data to {self.trainpath}")
            for i, train_data in enumerate(self.tqdm(self.train_data)):
                np.savez_compressed(self.trainpath / f"kfold_{i:d}.npz", **train_data)

            # Save testing data
            self.log.debug(f"Writing testing data to {self.testpath}")
            for i, test_data in enumerate(self.tqdm(self.test_data)):
                np.savez_compressed(self.testpath / f"kfold_{i:d}.npz", **test_data)
        return True, None
