from typing import TYPE_CHECKING, final, override

import numpy as np

from suPAErnova.steps import ModelStep
from suPAErnova.model_utils import losses
from suPAErnova.config.tf_autoencoder import (
    prev,
    optional,
    required,
    optional_params,
    required_params,
)

if TYPE_CHECKING:
    from suPAErnova.utils.typing import CFG
    from suPAErnova.models.tf_autoencoder import TFAutoencoder


@final
class TF_AutoEncoder(ModelStep):
    required = required
    optional = optional
    prev = prev
    required_params = required_params
    optional_params = optional_params

    def __init__(self, cfg: "CFG") -> None:
        super().__init__(cfg)

        # Model
        self.model_cls: type[TFAutoencoder]
        self.model: TFAutoencoder

        # Training Params
        self.epochs: int
        self.epochs_physical: int
        self.epochs_latent: int
        self.epochs_final: int
        self.validate_every_n: int

    @override
    def _setup(self):
        super()._setup()
        colourlaw = self.params["COLOURLAW"]
        if colourlaw is not None:
            _, CL = np.loadtxt(colourlaw, unpack=True)
            self.params["COLOURLAW"] = CL
        self.epochs_physical = self.params["EPOCHS_PHYSICAL"]
        self.epochs_latent = self.params["EPOCHS_LATENT"]
        self.epochs_final = self.params["EPOCHS_FINAL"]
        self.validate_every_n = self.params["VALIDATE_EVERY_N"]
        return (True, None)

    def train_model(self):
        ncoloumn_loss = 4
        training_losses = np.zeros((self.epochs, ncoloumn_loss))
        validation_losses = np.zeros((
            self.epochs // self.validate_every_n,
            ncoloumn_loss,
        ))
        testing_losses = np.zeros((
            self.epochs // self.validate_every_n,
            ncoloumn_loss,
        ))

        for epoch in range(self.epochs):
            is_best = False

            training_loss, training_loss_terms = train_step(
                self.model,
                optimizer,
                compute_apply_gradients_ae,
                epoch,
                nbatches,
                train_data,
            )

            # Get average loss over batches
            training_losses[epoch, 0] = epoch
            training_losses[epoch, 1:] = training_loss_terms

            if epoch % self.validate_every_n == 0:
                validation_loss, validation_loss_terms = test_step(
                    self.model,
                    validation_data,
                )
                validation_losses[epoch, 0] = epoch
                validation_losses[epoch, 1:] = validation_loss_terms

                testing_loss, testing_loss_terms = test_step(
                    self.model,
                    testing_data,
                )
                testing_losses[epoch, 0] = epoch
                testing_losses[epoch, 1:] = testing_loss_terms

                previous_validation_decrease = (
                    validation_iteration - best_validation_iteration
                )
                if validation_loss < validation_loss_min:
                    is_best = True
                    best_validation_iteration = validation_iteration
                    validation_loss_min = min(validation_loss_min, validation_loss)
                    self.model.save()
                validation_iteration += 1
            # Save model at last epoch
            if epoch == self.epochs - 1:
                self.model.save()

        return training_losses, validation_losses, testing_losses

    def train_physical(self) -> None:
        train_physical_params = {"train_stage": 0, "training": True}
        self.model = self.model_cls(self, train_physical_params)
        if self.global_cfg["VERBOSE"]:
            self.model.encoder.summary()
            self.model.decoder.summary()
        self.epochs = self.epochs_physical
        self.train_model()

    @override
    def _is_completed(self) -> bool:
        return False

    @override
    def _load(self) -> None:
        return None

    @override
    def _run(self):
        self.train_physical()
        return True, None

    @override
    def _result(self):
        return True, None
