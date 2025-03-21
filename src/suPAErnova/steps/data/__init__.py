from typing import TYPE_CHECKING, ClassVar, override
from pathlib import Path

import numpy as np
from numpy import typing as npt  # noqa: TC002
import pandas as pd
from astropy import cosmology as cosmo
import sncosmo
from pydantic import BaseModel, ConfigDict

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.data import DataStepConfig

if TYPE_CHECKING:
    from typing import TypeVar
    from logging import Logger
    from collections.abc import Iterable, Sequence

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.config import GlobalConfig

    SNeDataFrame = pd.DataFrame

    T = TypeVar("T", bound=np.generic)


class SNPAEData(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)  # pyright: ignore[reportIncompatibleVariableOverride]

    ind: npt.NDArray[np.int32]
    nspectra: npt.NDArray[np.int32]
    sn_name: npt.NDArray[np.str_]
    dphase: npt.NDArray[np.float32]
    redshift: npt.NDArray[np.float32]
    x0: npt.NDArray[np.float32]
    x1: npt.NDArray[np.float32]
    c: npt.NDArray[np.float32]
    MB: npt.NDArray[np.float32]
    hubble_residual: npt.NDArray[np.float32]
    luminosity_distance: npt.NDArray[np.float32]
    spectra_id: npt.NDArray[np.int64]
    phase: npt.NDArray[np.float32]
    wl_mask_min: npt.NDArray[np.float32]
    wl_mask_max: npt.NDArray[np.float32]
    amplitude: npt.NDArray[np.float32]
    sigma: npt.NDArray[np.float32]
    salt_flux: npt.NDArray[np.float32]
    wavelength: npt.NDArray[np.float32]
    mask: npt.NDArray[np.int32]
    time: npt.NDArray[np.float32]


class DataStep(SNPAEStep[DataStepConfig]):
    # Class Variables
    name: ClassVar["str"] = "data"

    def __init__(self, config: "DataStepConfig") -> None:
        # --- Superclass Variables ---
        self.options: DataStepConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variables ---
        # Required
        self.data_dir: Path
        self.meta: Path
        self.idr: Path
        self.mask: Path

        # Optional
        self.cosmological_model: cosmo.FlatLambdaCDM
        self.salt_model: sncosmo.SALT2Source | sncosmo.SALT3Source
        self.min_phase: float
        self.max_phase: float
        self.train_frac: float
        self.seed: int

        # --- Setup Variables ---
        self.rng: np.random.Generator

        # Output paths
        self.out_data: Path
        self.out_sne: Path
        self.out_train: Path
        self.out_test: Path

        # Train / Test split
        self.test_frac: float
        self.nkfold: int

        # --- Run Variables ---
        self.wavelength: npt.NDArray[np.float32]

        # Output objects
        self.sne: SNeDataFrame
        self.data: SNPAEData
        self.train_data: list[SNPAEData]
        self.test_data: list[SNPAEData]

        # Data Dimensions
        self.sn_dim: int
        self.phase_dim: int
        self.wl_dim: int

    @override
    def _setup(self) -> None:
        # --- Config Variables ---
        # Required
        self.data_dir = self.options.data_dir
        self.meta = self.options.meta
        self.idr = self.options.idr
        self.mask = self.options.mask

        # Optional
        # Get astropy.cosmology model associated with provided cosmological_model string
        self.cosmological_model = getattr(cosmo, self.options.cosmological_model)

        # Get sncosmo SALTSource associated with provided salt_model string
        #   If salt_model is a valid Path, pass it to the SALTSource as the modeldir
        salt_model = self.options.salt_model
        if isinstance(salt_model, Path):
            if "salt2" in str(salt_model):
                self.salt_model = sncosmo.SALT2Source(salt_model)
            elif "salt3" in str(salt_model):
                self.salt_model = sncosmo.SALT3Source(salt_model)
        else:
            self.salt_model = sncosmo.get_source(salt_model)

        self.min_phase = self.options.min_phase
        self.max_phase = self.options.max_phase
        self.train_frac = self.options.train_frac
        self.seed = self.options.seed

        # --- Computed Variables ---
        self.rng = np.random.default_rng(self.seed)

        # Output paths
        self.out_data = self.paths.out / "data.npz"
        self.out_sne = self.paths.out / "sne.pkl"
        self.out_train = self.paths.out / "train"
        self.out_train.mkdir(parents=True, exist_ok=True)
        self.out_test = self.paths.out / "test"
        self.out_test.mkdir(parents=True, exist_ok=True)

        # Train / Test split
        self.test_frac = 1 - self.train_frac
        self.nkfold = int(1 / self.test_frac)

    @override
    def _completed(self) -> bool:
        if not self.out_data.exists():
            self.log.debug(
                f"{self.__class__.__name__} has not completed as {self.out_data} does not exist"
            )
            return False
        if not self.out_sne.exists():
            self.log.debug(
                f"{self.__class__.__name__} has not completed as {self.out_sne} does not exist"
            )
            return False
        if not self.out_train.exists():
            self.log.debug(
                f"{self.__class__.__name__} has not completed as {self.out_train} does not exist"
            )
            return False
        if not self.out_test.exists():
            self.log.debug(
                f"{self.__class__.__name__} has not completed as {self.out_test} does not exist"
            )
            return False
        if len(list(self.out_train.iterdir())) == 0:
            self.log.debug(
                f"{self.__class__.__name__} has not completed as {self.out_train} does not contain any files"
            )
            return False
        if len(list(self.out_test.iterdir())) == 0:
            self.log.debug(
                f"{self.__class__.__name__} has not completed as {self.out_test} does not contain any files"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        pass

    @override
    def _run(self) -> None:
        # Create self.sne
        self.load_sne()
        self.calculate_salt_flux()
        self.prepare_data_arrays()
        self.split_train_test()

    @override
    def _result(self) -> None:
        self.log.debug(f"Saving SNe DataFrame to {self.out_sne}")
        self.sne.to_pickle(self.out_sne)

        self.log.debug(f"Saving data arrays to {self.out_data}")
        np.savez_compressed(self.out_data, **self.data.model_dump())

        self.log.debug(f"Saving training data arrays to {self.out_train}")
        for i, train_data in enumerate(self.train_data):
            np.savez_compressed(
                self.out_train / f"kfold_{i:d}.npz", **train_data.model_dump()
            )

        self.log.debug(f"Saving testing data arrays to {self.out_test}")
        for i, test_data in enumerate(self.test_data):
            np.savez_compressed(
                self.out_test / f"kfold_{i:d}.npz", **test_data.model_dump()
            )

    @override
    def _analyse(self) -> None:
        pass

    def load_sne(self) -> None:
        self.log.debug(f"Loading data from `meta` file: {self.meta}")
        sne_dtypes = {
            "id": str,
            "sn": str,
            "phase": float,
            "z": float,
            "MB": float,
            "x0": float,
            "x1": float,
            "c": float,
            "path": str,
            "hubble_resid": float,
        }
        sne_data = pd.read_csv(self.meta, header=0, dtype=sne_dtypes)
        # Update paths relative to self.meta
        sne_data.path = sne_data.path.apply(
            lambda path: str(
                self.paths.resolve_path(Path(path), relative_path=self.meta.parent)
            )
        )

        self.log.debug(f"Loading data from `idr` file: {self.idr}")
        dphase_dtypes = {
            "sn": str,
            "mjd": float,
            "dphase": float,
        }
        dphase = pd.read_csv(
            self.idr, sep="\\s+", names=["sn", "mjd", "dphase"], dtype=dphase_dtypes
        )
        sne_data = sne_data.merge(dphase, on="sn", how="left")

        self.log.debug(f"Loading data from `mask` file: {self.mask}")
        mask_dtypes = {
            "sn": str,
            "id": str,
            "flag": int,
            "wl_min": float,
            "wl_max": float,
        }
        mask = pd.read_csv(
            self.mask,
            sep="\\s+",
            names=["sn", "id", "flag", "wl_min", "wl_max"],
            dtype=mask_dtypes,
        )
        mask.id = mask.sn + "_" + mask.id
        sne_data = sne_data.merge(mask, on=["sn", "id"], how="left")

        self.log.debug("Merging SNe data")
        # Split data into two dataframes

        # A SN dataframe which contains one row per SN, and the following columns
        sne_cols = ["sn", "MB", "x0", "x1", "c", "z", "hubble_resid", "mjd", "dphase"]
        sne = sne_data[sne_cols].drop_duplicates().reset_index(drop=True)

        # A dataframe which contains one row per spectra, and the following columns
        # Note that we keep the sn column so that we can match each spectra with their SN
        spec_cols = [
            "sn",
            "id",
            "phase",
            "path",
            "flag",
            "wl_min",
            "wl_max",
        ]
        spectra = sne_data[spec_cols].reset_index(drop=True)

        self.log.debug(
            f"Cutting spectra with phases outside the range {self.min_phase} <= phase <= {self.max_phase}",
        )
        self.log.debug(f"Number of spectra before phase-cut: {len(spectra)}")
        spectra = spectra[spectra.phase.between(self.min_phase, self.max_phase)]
        self.log.debug(f"Number of spectra after phase-cut: {len(spectra)}")

        self.log.debug("Loading spectra data")
        spectra_dtype = {"wave": float, "flux": float, "sigma": float}
        spectra["data"] = [
            pd.read_csv(spec.path, dtype=spectra_dtype)
            for _, spec in spectra.iterrows()
        ]

        self.log.debug("Linking spectra to their associated SNe")
        sne["spectra"] = sne.sn.apply(
            lambda sn_name: spectra[spectra.sn == sn_name].reset_index(drop=True),
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
        #       wl_min: float     = Min wavelength of spectra
        #       wl_max: float     = Max wavelength of spectra
        #       data:           DataFrame = Spectral data with columns:
        #
        #           wave:  Series[float]  = wavelength in AA
        #           flux:  Series[float]  = flux
        #           sigma: Series[float]  = flux error

        self.sne = sne

    def calculate_salt_flux(self) -> None:
        def get_salt_flux(
            wavelength: "npt.NDArray[np.float32]",
            tobs: float = 0.0,
            z: float = 0.0,
            x0: float = 1.0,
            x1: float = 0.0,
            c: float = 0.0,
            zref: float = 0.05,
        ) -> "npt.NDArray[np.float32]":
            self.salt_model.set(x0=x0, x1=x1, c=c)
            return (
                self.salt_model.flux(phase=tobs, wave=wavelength)
                * (
                    (
                        self.cosmological_model.luminosity_distance(z)
                        / self.cosmological_model.luminosity_distance(zref)
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

    def prepare_data_arrays(self) -> None:
        # Each element of data is a 3D Array of shape (sn_dim x phase_dim x data_dim) where:
        #   sn_dim = Number of SNe
        #   phase_dim = Maximum number of observations for any given SN (padded if needed)
        #   data_dim = Length of datatype

        # --- Get Dimensions ---
        self.sn_dim = len(self.sne)
        self.log.debug(f"Number of SNe: {self.sn_dim}")

        # Maximum number of observations for any given SN
        nspectra_per_sn = np.array(
            [len(spectra) for spectra in self.sne["spectra"]],
        )
        self.phase_dim = nspectra_per_sn.max()
        self.log.debug(
            f"Maximum number of observations for any given SN: {self.phase_dim}",
        )

        # Wavelength grid
        # Since all spectra share the same wavelength grid
        # Just get the wavelength grid of the first spectrum
        self.wavelength = self.sne["spectra"][0]["data"][0]["wave"].to_numpy()
        self.wl_dim = len(self.wavelength)
        self.log.debug(f"Length of wavelength grid: {self.wl_dim}")

        # Allows for filling an array with padding
        phase_axis = nspectra_per_sn.copy()
        phase_axis.fill(self.phase_dim)

        # --- Get Parameters ---
        data = {}

        # Given an array of shape sn_dim by N <= phase_dim
        # Create an array of shape sn_dim by phase_dim, padding if needed
        def pad(
            arr: "Iterable[Sequence[T | npt.NDArray[T]]]",
            padding: "T | npt.NDArray[T]",
        ) -> "npt.NDArray[T]":
            if isinstance(padding, np.ndarray):
                padded_arr: npt.NDArray[T] = np.full(
                    (self.sn_dim, self.phase_dim, *padding.shape),
                    padding,
                )
            else:
                padded_arr = np.full((self.sn_dim, self.phase_dim), padding)
            for i, row in enumerate(arr):
                row_length = len(row)
                padded_arr[i, :row_length] = row
            return padded_arr

        # Given a list of value-per-row of length sn_dim
        # Fill each row with phase_dim repeats of that row's value
        def fill_rows(values: "npt.NDArray[T]") -> "npt.NDArray[T]":
            return np.repeat(values, phase_axis).reshape((self.sn_dim, self.phase_dim))

        # Index of each SNe
        data["ind"] = fill_rows(np.array(range(self.sn_dim)))

        # Number of spectra per SNe
        data["nspectra"] = fill_rows(nspectra_per_sn)

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
            data[data_key] = fill_rows(
                self.sne[sne_key].to_numpy(),
            )

        data["luminosity_distance"] = self.cosmological_model.luminosity_distance(
            data["redshift"],
        ).value

        # Get Parameters from spectra
        spectra_params = {
            "spectra_id": ("id", np.str_("")),
            "phase": ("phase", np.float32(-100.0)),
            "wl_mask_min": ("wl_min", np.float32(-1.0)),
            "wl_mask_max": ("wl_max", np.float32(-1.0)),
        }

        for data_key, (spectra_key, padding) in spectra_params.items():
            data[data_key] = pad(
                [spectra[spectra_key].to_numpy() for spectra in self.sne["spectra"]],
                padding,
            )

        # Get spectral data parameters
        spectral_data_params = {
            "amplitude": ("flux", np.zeros(self.wl_dim)),
            "sigma": ("sigma", np.zeros(self.wl_dim)),
            "salt_flux": ("salt_flux", np.zeros(self.wl_dim)),
        }

        for data_key, (spectral_data_key, padding) in spectral_data_params.items():
            data[data_key] = pad(
                [
                    [
                        spectral_data[spectral_data_key].to_numpy()
                        for spectral_data in spectra["data"]
                    ]
                    for spectra in self.sne["spectra"]
                ],
                padding,
            )

        data["wavelength"] = pad(
            [
                [self.wavelength for _ in spectra["data"]]
                for spectra in self.sne.spectra
            ],
            np.zeros(self.wl_dim),
        )

        # Ensure everything has the right number of axes
        for k, v in data.items():
            if len(v.shape) == 2:
                data[k] = v[..., np.newaxis]

        # Create a mask of wavelength outside of the wavelength limits
        data["mask"] = (data["wl_mask_min"] <= data["wavelength"]) & (
            data["wavelength"] <= data["wl_mask_max"]
        )

        # Mask any huge laser lines, Na D (5674 - 5692A)
        # TODO: Make these options
        # these are large jumps in flux, localized over a few wavelength bins
        wavelength_bin_start = 5000.0
        wavelength_bin_end = 8000.0
        laser_width = 2  # in units of wavelength bins
        laser_height = 0.4  # fractional increase in amplitude over neighbours to be considered laser

        wave_mask = (wavelength_bin_start <= data["wavelength"]) & (
            data["wavelength"] <= wavelength_bin_end
        )
        amplitude = np.zeros(data["amplitude"].shape)
        amplitude[wave_mask] = data["amplitude"][wave_mask]

        wave_mask_min = (wavelength_bin_start - laser_width <= data["wavelength"]) & (
            data["wavelength"] <= wavelength_bin_end - laser_width
        )
        min_amplitude = np.zeros(data["amplitude"].shape)
        min_amplitude[wave_mask] = data["amplitude"][wave_mask_min]

        wave_mask_max = (wavelength_bin_start + laser_width <= data["wavelength"]) & (
            data["wavelength"] <= wavelength_bin_end + laser_width
        )
        max_amplitude = np.zeros(data["amplitude"].shape)
        max_amplitude[wave_mask] = data["amplitude"][wave_mask_max]

        smooth_amplitude = 0.5 * (min_amplitude + max_amplitude)

        laser_mask = (amplitude - smooth_amplitude) > laser_height
        data["mask"] &= ~laser_mask

        # --- Finalise Data ---
        # Rescale phase to time such that:
        #   time = 0 -> phase = min_phase
        #   time = 1 -> phase = max_phase
        time_mask = data["phase"] > -100
        data["time"] = data["phase"].copy()
        data["time"][time_mask] = (data["time"][time_mask] - self.min_phase) / (
            self.max_phase - self.min_phase
        )

        # Remove negative amplitude from unmasked amplitudes
        data["amplitude"][data["mask"]] = np.clip(
            data["amplitude"][data["mask"]],
            0,
            np.inf,
        )

        # Scale observed uncertainty to account for fitting degrees of freedom, and an error floor
        data["sigma"] = 1.4 * data["sigma"] + 4e-10

        self.data = SNPAEData.model_validate(data)

    def split_train_test(self) -> None:
        self.train_data = []
        self.test_data = []

        # Train test split
        ind_split = int(self.sn_dim * self.train_frac)

        # Select train_frac for training, the rest for testing
        inds = np.arange(0, self.sn_dim)
        self.rng.shuffle(inds)

        # Split into k cross validation sets
        for kfold in range(self.nkfold):
            inds_k = np.roll(inds, kfold * inds.shape[0] // self.nkfold)

            inds_train = inds_k[:ind_split]
            inds_test = inds_k[ind_split:]

            self.train_data.append(
                SNPAEData.model_validate({
                    key: val[inds_train, :, :] if val.ndim == 3 else val[inds_train, :]
                    for key, val in self.data.model_dump().items()
                })
            )
            self.test_data.append(
                SNPAEData.model_validate({
                    key: val[inds_test, :, :] if val.ndim == 3 else val[inds_test, :]
                    for key, val in self.data.model_dump().items()
                })
            )
        n_train_sn = self.train_data[0].amplitude.shape[0]
        n_test_sn = self.test_data[0].amplitude.shape[0]
        self.log.debug(
            f"n_train_sn: {n_train_sn} ({100 * n_train_sn / self.sn_dim}%) + n_test_sn: {n_test_sn} ({100 * n_test_sn / self.sn_dim}%) = {n_train_sn + n_test_sn} ({100 * (n_train_sn + n_test_sn) / self.sn_dim}%)",
        )


DataStep.register_step()
