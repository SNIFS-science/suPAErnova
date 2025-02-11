from typing import TYPE_CHECKING, Literal

import pandas as pd
import matplotlib as mpl
from matplotlib import (
    cm,
    pyplot as plt,
)

if TYPE_CHECKING:
    from pathlib import Path

    from suPAErnova.steps import Data


def _plot_spectra(self: "Data", sn: pd.DataFrame, plotpath: "Path") -> None:
    name = sn["sn"].to_numpy()[0]
    outpath = plotpath / f"{name}.svg"
    if not self.force and outpath.exists():
        self.log.debug(f"{name} plot already exists at {outpath}")
        return

    spectra = pd.concat(sn["spectra"].tolist())
    phases = spectra["phase"]

    norm = mpl.colors.Normalize(phases.min(), phases.max())
    cmap = cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    cmap.set_array([])

    plt.figure(figsize=(12, 4))
    for _, spec in spectra.iterrows():
        data = spec["data"]
        wave = data["wave"]
        flux = data["flux"]
        sigma = data["sigma"]
        plt.errorbar(wave, flux, yerr=sigma, color=cmap.to_rgba(spec["phase"]))

    plt.colorbar(cmap, label="Phase", ax=plt.gca())
    plt.xlabel("Rest-frame wavelength (AA)")
    plt.ylabel("Normalized flux")
    plt.title(name)

    plt.savefig(outpath)
    # Clear plots from memory
    plt.cla()
    plt.clf()
    plt.close("all")


def plot_spectra(
    self: "Data",
    spectra_to_plot: str | list[str] | Literal[True],
) -> None:
    self.log.info("Plotting Spectra")
    if not isinstance(self.sne, pd.DataFrame):
        self.log.error("Tried to plot without first loading SNe")
        return
    sn_names = self.sne["sn"]
    if isinstance(spectra_to_plot, str):
        spectra_to_plot = [spectra_to_plot]
    # If not a list, then self.spectra_to_plot = True, so use all SNe
    if isinstance(spectra_to_plot, list):
        names = sn_names[sn_names.isin(spectra_to_plot)]
        for missing_name in filter(
            lambda x: x not in names.to_numpy(),
            spectra_to_plot,
        ):
            self.log.warning(f"SN {missing_name} not found")
    if len(sn_names) == 0:
        self.log.warning("No SN to plot!")
        return
    for sn in self.tqdm(sn_names, desc="Plotting", total=len(sn_names)):
        self.log.debug(f"Plotting {sn}")
        _plot_spectra(
            self,
            self.sne[self.sne["sn"] == sn],
            self.plotpath,
        )


ANALYSES = {
    "PLOT_SPECTRA": plot_spectra,
}
