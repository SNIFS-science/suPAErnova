from matplotlib import pyplot as plt

from suPAErnova.analysis import (
    data as DATA,
    tf_autoencoder as TF_AUTOENCODER,
)

plt.set_loglevel("warning")

__all__ = ["DATA", "TF_AUTOENCODER"]
