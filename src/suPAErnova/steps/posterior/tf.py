# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    cast,
    override,
)

import keras
from tensorflow import keras as ks

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.pae.tf import TensorCompatible
    from suPAErnova.steps.nflow.tf import TFNFlowModel
    from suPAErnova.steps.posterior.model import PosteriorModel
    from suPAErnova.configs.steps.posterior.tf import TFPosteriorModelConfig

    # === Custom Types ===
    PosteriorInputs = Any
    PosteriorOutputs = Any


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPosteriorModel(ks.Model):
    def __init__(
        self,
        config: "PosteriorModel[TFPosteriorModel, TFPosteriorModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(
            *args, name=f"{config.name.split()[-1]}PosteriorModel", **kwargs
        )
        # --- Config ---
        options = cast("TFPosteriorModelConfig", config.options)
        self.log: "Logger" = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.debug: bool = options.debug
        self.nflow: "TFNFlowModel" = cast("TFNFlowModel", config.nflow.model)

        # --- Training ---
        self._epoch: int = 0
        self.built: bool = False
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

    @override
    def call(
        self,
        inputs: "PosteriorInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "PosteriorOutputs":
        training = False if training is None else training
        return None

    def train_model(
        self,
        *,
        savepath: "Path | None" = None,
    ) -> ks.callbacks.History:
        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        if savepath is not None:
            # --- Backup & Restore ---
            # Backup checkpoints each epoch and restore if training got cancelled midway through
            if not self.force:
                backup_dir = savepath / "backups"
                backup_callback = ks.callbacks.BackupAndRestore(str(backup_dir))
                callbacks.append(backup_callback)

            # --- Model Checkpoint ---
            checkpoint_callback = ks.callbacks.ModelCheckpoint(
                str(savepath / self.weights_path),
                save_best_only=self.save_best,
                save_weights_only=True,
                # verbose=0,
            )
            callbacks.append(checkpoint_callback)

        # --- Terminate on NaN ---
        # Terminate training when a NaN loss is encountered
        callbacks.append(ks.callbacks.TerminateOnNaN())

        self.build_model()

        # === Train ===
        self._epoch = 0
        return self.fit(
            callbacks=callbacks,
            # verbose=0,
        )

    def build_model(
        self,
    ) -> None:
        if not self.built:
            self.compile(
                run_eagerly=self.debug,
            )
            self((self.data), training=False)

            self.log.debug("Trainable variables:")
            for var in self.trainable_variables:
                self.log.debug(f"{var.name}: {var.shape}")
            self.summary(print_fn=self.log.debug)  # Will show number of parameters

            self.built = True

    def save_checkpoint(self, savepath: "Path") -> None:
        self.save_weights(savepath / self.weights_path)
        self.save(savepath / self.model_path)

    def load_checkpoint(self, loadpath: "Path") -> None:
        self.build_model()
        self.load_weights(loadpath / self.weights_path)
