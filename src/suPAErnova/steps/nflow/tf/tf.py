# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    override,
)

import tensorflow as tf
from tensorflow import keras as ks

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.nflow.model import NFlowModel
    from suPAErnova.configs.steps.pae.tf.tf import TFPAEModelConfig


class TFNFlowModel(ks.Model):
    def __init__(
        self,
        config: "NFlowModel[TFNFlowModel]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=f"{config.name.split()[-1]}PAEModel", **kwargs)
        # --- Config ---
        options = cast("TFPAEModelConfig", config.options)
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
