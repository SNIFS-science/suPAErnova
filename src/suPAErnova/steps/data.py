from typing import TYPE_CHECKING, final, override

import numpy as np
import pandas as pd

from suPAErnova.steps import Step, callback
from suPAErnova.config.data import optional, required

if TYPE_CHECKING:
    from typing import Literal, TypeVar
    from pathlib import Path
    from collections.abc import Iterable, Sequence

    from astropy import cosmology as cosmo
    import sncosmo
    import numpy.typing as npt

    from suPAErnova.utils.typing import CFG

    T = TypeVar("T", bound=np.generic)


@final
class Data(Step):
    required = required
    optional = optional

    def __init__(self, cfg: "CFG") -> None:
        super().__init__(cfg)

        # Default
        self.datapath = self.outpath / "data.npz"
        self.trainpath = self.outpath / "train"
        if not self.trainpath.exists():
            self.trainpath.mkdir(parents=True)
        self.testpath = self.outpath / "test"
        if not self.testpath.exists():
            self.testpath.mkdir(parents=True)

        # Load from config file
        # Required
        self.metapath: Path = self.opts["META"]
        self.idrpath: Path = self.opts["IDR"]
        self.maskpath: Path = self.opts["MASK"]

        # Optional
        # SALT Setup
        self.cosmo: cosmo.FlatLambdaCDM = self.opts["COSMO"]
        self.snmodel: sncosmo.SALT2Source | sncosmo.SALT3Source = self.opts["SALT"]
        self.log.debug(f"SALT model parameters are: {self.snmodel.param_names}")

        # Quality Cuts
        self.min_phase = self.opts["MIN_PHASE"]
        self.max_phase = self.opts["MAX_PHASE"]

        # Train and Test split
        self.train_frac = self.opts["TRAIN_FRAC"]
        self.test_frac = 1 - self.train_frac
        self.nkfold = int(1.0 / self.test_frac)
        self.seed = self.opts["SEED"]
        self.rng = np.random.default_rng(self.seed)

        # Generate in self._run()
        # Lengths
        self.n_sn: int | None = None
        self.n_timemax: int | None = None
        self.n_timemin: int | None = None
        self.n_wavelength: int | None = None

        # Data Products
        self.sne: pd.DataFrame | None = None
        self.data: CFG | None = None
        self.train_data: list[CFG] = []
        self.test_data: list[CFG] = []

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
        sne = sn_data[sne_cols].drop_duplicates().reset_index(drop=True)

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
        spectra = sn_data[spec_cols].reset_index(drop=True)

        self.log.debug(
            f"Cutting spectra with phases outside the range {self.min_phase} <= phase <= {self.max_phase}",
        )
        self.log.debug(f"Number of spectra before phase-cut: {len(spectra)}")
        spectra = spectra[spectra["phase"].between(self.min_phase, self.max_phase)]
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
        sne["spectra"] = sne["sn"].apply(
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
        if not isinstance(self.sne, pd.DataFrame):
            self.log.error("Tried to generate SALT data without first loading SNe")
            return

        def get_salt_flux(
            wavelength: "float | Sequence[float]",
            tobs: float = 0.0,
            z: float = 0.0,
            x0: float = 1.0,
            x1: float = 0.0,
            c: float = 0.0,
            t0: float = 0.0,
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
            for _, spectra in sn["spectra"].iterrows():
                spectra["data"]["salt_flux"] = get_salt_flux(
                    spectra["data"]["wave"].to_numpy(),
                    tobs=spectra["phase"],
                    z=sn["z"],
                    x0=sn["x0"],
                    x1=sn["x1"],
                    c=sn["c"],
                )

    @callback
    def transform_data(self) -> None:
        if not isinstance(self.sne, pd.DataFrame):
            self.log.error("Tried to generate data without first loading SNe")
            return

        # Each element of data is a 3D Array of shape (n_sn x n_timemax x n_data) where:
        #   n_sn = Number of SNe
        #   n_timemax = Maximum number of observations for any given SN (padded if needed)
        #   n_data = Length of datatype

        # --- Get Dimensions ---
        # Number of SNe
        self.n_sn = n_sn = len(self.sne)
        self.log.debug(f"Number of SNe: {n_sn}")

        # Maximum number of observations for any given SN
        nspectra_per_sn = np.array([len(spectra) for spectra in self.sne["spectra"]])
        self.n_timemax = n_timemax = nspectra_per_sn.max()
        self.n_timemin = n_timemin = nspectra_per_sn.min()
        self.log.debug(f"Maximum number of observations for any given SN: {n_timemax}")
        # Allows for filling an array with padding
        time_axis = nspectra_per_sn.copy()
        time_axis.fill(n_timemax)

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
                    (n_sn, n_timemax, *padding.shape),
                    padding,
                    dtype=object,
                )
            else:
                padded_arr = np.full((n_sn, n_timemax), padding, dtype=object)
            for i, row in enumerate(arr):
                row_length = len(row)
                padded_arr[i, :row_length] = row
            return padded_arr

        # Given a list of value-per-row of length n_sn
        # Fill each row with n_timemax repeats of that row's value
        def fill_rows(values: "npt.NDArray[T]") -> "npt.NDArray[T]":
            return np.repeat(values, time_axis).reshape((n_sn, n_timemax))

        # Index of each SNe
        self.data["ind"] = fill_rows(np.array(range(n_sn)))

        # Number of spectra per SNe
        self.data["nspectra"] = fill_rows(nspectra_per_sn)

        # Wavelength grid
        # Since all spectra share the same wavelength grid
        # Just get the wavelength grid of the first spectrum
        wavelength = self.sne["spectra"][0]["data"][0]["wave"]
        self.n_wavelength = n_wavelength = len(wavelength)
        self.log.debug(f"Length of wavelength grid: {n_wavelength}")

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
            self.data[data_key] = fill_rows(self.sne[sne_key].to_numpy())

        self.data["luminosity_distance"] = self.cosmo.luminosity_distance(
            self.data["redshift"],
        ).value

        # Get Parameters from spectra
        spectra_params = {
            "spectra_id": ("id", np.str_("")),
            "phase": ("phase", np.int64(-1)),
            "wl_mask_min": ("wavelength_min", np.float64(-1)),
            "wl_mask_max": ("wavelength_max", np.float64(-1)),
        }

        for data_key, (spectra_key, padding) in spectra_params.items():
            self.data[data_key] = pad(
                [spectra[spectra_key].to_numpy() for spectra in self.sne["spectra"]],
                padding,
            )

        # Get spectral data parameters
        spectral_data_params = {
            "flux": ("flux", np.zeros(n_wavelength)),
            "sigma": ("sigma", np.zeros(n_wavelength)),
            "salt_flux": ("salt_flux", np.zeros(n_wavelength)),
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
                [wavelength.to_numpy() for _ in spectra["data"]]
                for spectra in self.sne["spectra"]
            ],
            np.zeros(n_wavelength),
        )

    @callback
    def calculate_wavelength_mask(self) -> None:
        if self.data is None:
            self.log.error(
                "Tried to calculate wavelength mask without first transforming data",
            )
            return
        # Create a mask of wavelength outside of the wavelength limits
        self.data["mask"] = (
            self.data["wl_mask_min"][..., np.newaxis] <= self.data["wavelength"]
        ) & (self.data["wavelength"] <= self.data["wl_mask_max"][..., np.newaxis])

    @callback
    def calculate_laser_line_mask(self) -> None:
        if self.data is None:
            self.log.error(
                "Tried to calculate wavelength mask without first transforming data",
            )
            return

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
        if self.data is None:
            self.log.error("Tried to split data without first generating data")
            return
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
            and self.trainpath.exists()
            and len(list(self.trainpath.iterdir())) > 0
            and self.testpath.exists()
            and len(list(self.trainpath.iterdir())) > 0
        )

    @override
    def _load(self) -> None:
        if self.datapath.exists():
            # TODO: self.data -> self.sne
            #       or save self.sne to file
            self.load_data()
            self.calculate_salt_flux()
            # Load data from files
            self.data = np.load(self.datapath, allow_pickle=True)
            self.train_data = [
                np.load(train_data, allow_pickle=True)
                for train_data in self.trainpath.iterdir()
                if train_data.is_file()
            ]
            self.test_data = [
                np.load(test_data, allow_pickle=True)
                for test_data in self.testpath.iterdir()
                if test_data.is_file()
            ]

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
        # Provides self.train and self.test
        self.split_train_test()
        return True, None

    @override
    def _result(self):
        if self.force or not self._is_completed():
            # Save data
            if self.data is None:
                return False, "Data has not been generated"
            self.log.debug(f"Writing generated data to {self.datapath}")
            np.savez_compressed(self.datapath, **self.data)

            # Save training data
            if len(self.train_data) == 0:
                return False, "Train data has not been generated"
            self.log.debug(f"Writing training data to {self.trainpath}")
            for i, train_data in enumerate(self.tqdm(self.train_data)):
                np.savez_compressed(self.trainpath / f"kfold_{i:d}.npz", **train_data)

            # Save testing data
            if len(self.test_data) == 0:
                return False, "Test data has not been generated"
            self.log.debug(f"Writing testing data to {self.testpath}")
            for i, test_data in enumerate(self.tqdm(self.test_data)):
                np.savez_compressed(self.testpath / f"kfold_{i:d}.npz", **test_data)
        return True, None
