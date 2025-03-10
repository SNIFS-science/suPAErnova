# Copyright 2025 Patrick Armstrong
"""Data step analysis functions."""

from typing import TYPE_CHECKING

import pandas as pd
import matplotlib as mpl
from matplotlib import (
    cm,
    pyplot as plt,
)

if TYPE_CHECKING:
    from typing import Literal
    from pathlib import Path

    from suPAErnova.steps import Data


def _plot_spectra(self: "Data", sn: pd.DataFrame, plotpath: "Path") -> None:
    name = sn["sn"].to_numpy()[0]
    svg_outpath = plotpath / f"{name}.svg"
    png_outpath = plotpath / f"{name}.png"
    if not self.force and svg_outpath.exists() and png_outpath.exists():
        self.log.debug(f"{name} plot already exists at {svg_outpath} / {png_outpath}")
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

    plt.savefig(png_outpath)
    plt.savefig(svg_outpath)
    # Clear plots from memory
    plt.cla()
    plt.clf()
    plt.close("all")


def plot_spectra(
    self: "Data",
    spectra_to_plot: "str | list[str] | Literal[True]",
) -> None:
    """Plot SN spectra.

    Args:
        self (Data): The Data step to plot
        spectra_to_plot (str | list[str] | Literal[True]): Which spectra to plot:
            spectra_to_plot (str): The name of a single spectrum to plot
            spectra_to_plot (list[str]): Names of each spectrum to plot
            spectra_to_plot (Literal[True]): Plot every spectrum
    """
    self.log.info("Plotting Spectra")
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
