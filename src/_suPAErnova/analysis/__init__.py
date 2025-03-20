# Copyright 2025 Patrick Armstrong
"""Step-specific analysis function."""

from matplotlib import pyplot as plt

from suPAErnova.analysis import (
    pae as PAEAnalysis,
    data as DATAAnalysis,
)
from suPAErnova.analysis.pae import tf_pae as TF_PAEAnalysis

plt.set_loglevel("warning")

__all__ = ["DATAAnalysis", "PAEAnalysis", "TF_PAEAnalysis"]
