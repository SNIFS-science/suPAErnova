import time
from typing import TYPE_CHECKING, Any, final, override

import keras as ks
import numpy as np
import tensorflow as tf

from suPAErnova.steps import ModelStep, callback
from suPAErnova.model_utils import (
    tf_losses as losses,
    tf_optimisers as optimisers,
    tf_schedulers as schedulers,
)
from suPAErnova.config.tf_autoencoder import (
    prev,
    optional,
    required,
    optional_params,
    required_params,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorflow.python.types.core import PolymorphicFunction

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

        # Default
        self.dtype = ks.backend.floatx()

        # Model
        self.model_cls: type[TFAutoencoder]
        self.model: TFAutoencoder

        # Training Params
        self.latent_dim: int
        self.epochs: int
        self.epochs_colour: int
        self.epochs_latent: int
        self.epochs_amplitude: int
        self.epochs_all: int
        self.validate_every_n: int
        self.min_train_redshift: float
        self.max_train_redshift: float
        self.min_train_phase: int
        self.max_train_phase: int
        self.split_training: bool
        self.noise_scale: float
        self.mask_frac: float
        self.sigma_time: int

        # Network Params
        self.lr: float
        self.lr_deltat: float
        self.lr_decay_rate: float
        self.lr_decay_steps: int
        self.weight_decay_rate: float
        self.batch_size: int

        self.scheduler: Callable[
            [float, dict[str, Any]],
            ks.optimizers.schedules.LearningRateSchedule | float,
        ]
        self.optimiser: Callable[
            [ks.optimizers.schedules.LearningRateSchedule | float, dict[str, Any]],
            ks.optimizers.Optimizer,
        ]
        # Data Params
        self.kfold: list[int]
        self.train_data: list[CFG]
        self.test_data: list[CFG]
        self.validation_data: list[CFG]
        self.training_data: tuple[CFG, CFG, CFG]
        self.validation_frac: float

    @override
    def _setup(self):
        super()._setup()
        colourlaw = self.params["COLOURLAW"]
        if colourlaw is not None:
            _, CL = np.loadtxt(colourlaw, unpack=True)
            self.params["COLOURLAW"] = CL
        self.latent_dim = self.params["LATENT_DIM"]
        self.epochs_colour = self.params["EPOCHS_COLOUR"]
        self.epochs_latent = self.params["EPOCHS_LATENT"]
        self.epochs_amplitude = self.params["EPOCHS_AMPLITUDE"]
        self.epochs_all = self.params["EPOCHS_ALL"]
        self.validate_every_n = self.params["VALIDATE_EVERY_N"]
        self.min_train_redshift = self.params["MIN_TRAIN_REDSHIFT"]
        self.max_train_redshift = self.params["MAX_TRAIN_REDSHIFT"]
        self.min_train_phase = self.params["MIN_TRAIN_PHASE"]
        self.max_train_phase = self.params["MAX_TRAIN_PHASE"]
        self.split_training = self.params["SPLIT_TRAINING"]
        self.noise_scale = self.params["NOISE_SCALE"]
        self.mask_frac = self.params["MASK_FRAC"]
        self.sigma_time = self.params["SIGMA_TIME"]

        self.lr = self.params["LR"]
        self.lr_deltat = self.params["LR_DELTAT"]
        self.lr_decay_rate = self.params["LR_DECAY_RATE"]
        self.lr_decay_steps = self.params["LR_DECAY_STEPS"]
        self.weight_decay_rate = self.params["WEIGHT_DECAY_RATE"]
        self.batch_size = self.params["BATCH_SIZE"]

        self.optimiser = optimisers.optimiser[self.params["OPTIMISER"]]
        self.scheduler = schedulers.scheduler[self.params["SCHEDULER"]]

        # Data Settings
        kfold = self.params["KFOLD"]
        if kfold is True:
            self.kfold = list(range(len(self.data.train_data)))
        else:
            self.kfold = kfold
        self.train_data = [self.prep_data(self.data.train_data[i]) for i in self.kfold]
        self.test_data = [self.prep_data(self.data.test_data[i]) for i in self.kfold]
        self.validation_frac = self.params["VALIDATION_FRAC"]
        if self.validation_frac > 0:
            split_train_data = []
            split_validation_data = []
            for train_data, _validation_data in zip(
                self.train_data,
                self.validation_data,
                strict=True,
            ):
                split_train_data.append({})
                split_validation_data.append({})
                num_samples = train_data["flux"].shape[0]
                num_validation_samples = np.ceil(num_samples * self.validation_frac)
                for k, v in train_data.items():
                    split_train_data[-1][k] = v[:-num_validation_samples]
                    split_validation_data[-1][k] = v[-num_validation_samples]
            self.train_data = split_train_data
            self.validation_data = split_validation_data
        else:
            self.validation_data = self.test_data

        return (True, None)

    def prep_data(self, data: "CFG") -> "CFG":
        redshift_mask = (data["redshift"] >= self.min_train_redshift) & (
            data["redshift"] <= self.max_train_redshift
        )
        phase_mask = (data["phase"] >= self.min_train_phase) & (
            data["phase"] <= self.max_train_phase
        )
        data["mask_train"] = redshift_mask & phase_mask
        return data
        # TODO: Add a `get_twins_mask` callback example

    def train_step(self, epoch: int, loss: "PolymorphicFunction"):
        training_loss = 0.0
        training_loss_terms = [0.0, 0.0, 0.0]
        train_data, test_data, validation_data = self.training_data

        # shuffle indices each epoch for batches
        # batch feeding can be improved, but the various types of specialized masks/dropout
        # are easy to implement in this non-optimized fashion, and the dataset is small
        rng = np.random.default_rng(epoch)
        inds = np.arange(train_data["flux"].shape[0])
        rng.shuffle(inds)
        inds = inds.reshape(-1, self.batch_size)
        nbatches = inds.shape[0]

        # Add noise during training drawn from observational uncertainty
        if self.noise_scale > 0:
            dflux = self.noise_scale * np.abs(
                rng.normal(0, train_data["sigma"]),
            )
        else:
            dflux = np.zeros(train_data["mask"].shape)

        # Mask certain spectra
        # Start with an empty mask which lets everything through
        dmask = np.ones(train_data["mask"].shape)

        # Mask a random fraction of spectra
        if self.mask_frac > 0:
            # First get the number of unmasked spectra for each SN
            n_unmasked_spectra_per_SN = np.sum(train_data["mask"], axis=1).astype(
                np.int32,
            )

            # Then get the number of those unmasked spectra to mask
            n_spectra_to_mask_per_SN = (
                self.mask_frac * n_unmasked_spectra_per_SN
            ).astype(np.int32)
            # For each supernova:
            for sn in range(dmask.shape[0]):
                # Get the number of unmasked spectra
                n_unmasked_spectra = n_unmasked_spectra_per_SN[sn][0]
                # And the number of enmased spectra to mask
                n_spectra_to_mask = n_spectra_to_mask_per_SN[sn][0]

                # Mask the first n_spectra_to_mask
                dmask[sn, :n_spectra_to_mask] = 0.0
                # Shuffle the position of all originally unmasked spectra
                # So that the newly masked spectra are randomly shuffled throughout
                dmask[sn : sn + 1, :n_unmasked_spectra] = dmask[
                    sn : sn + 1,
                    rng.random(n_unmasked_spectra).argsort(),
                ]

        dmask *= ~train_data["mask_train"]
        # Vary phase by observational uncertainty
        if self.sigma_time < 0:
            dtime = np.zeros(train_data["time"].shape[0])
        elif self.sigma_time == 0:
            dtime = rng.normal(0, train_data["dphase"] / 50)
        else:
            dtime = rng.normal(
                0,
                self.sigma_time / 50,
                size=(train_data["times"].shape[0], 1, 1),
            )

        # Loop over batches
        for batch in range(nbatches):
            inds_batch = sorted(inds[batch])

            # Flux + flux variations
            batch_flux = train_data["flux"][inds_batch] + dflux[inds_batch]
            # Time + time variations
            batch_time = train_data["time"][inds_batch] + dtime[inds_batch]
            # Flux error
            batch_sigma = train_data["sigma"][inds_batch]
            # Mask + mask variations
            batch_mask = train_data["mask"][inds_batch] * dmask[inds_batch]

            # Transform each array into the default tensor float dtype
            # We wait till this point to do it so that we don't lose any precision prior to downscaling

            batch_flux = tf.convert_to_tensor(batch_flux, dtype=self.dtype)
            batch_time = tf.convert_to_tensor(batch_time, dtype=self.dtype)
            batch_sigma = tf.convert_to_tensor(batch_sigma, dtype=self.dtype)
            batch_mask = tf.convert_to_tensor(batch_mask, dtype=self.dtype)

            # Train batch
            batch_training_loss, batch_training_loss_terms = loss(
                self.model,
                batch_flux,
                batch_time,
                batch_sigma,
                batch_mask,
            )

            training_loss += batch_training_loss
            for i, b_loss in enumerate(batch_training_loss_terms):
                training_loss_terms[i] += b_loss
        training_loss_terms = [loss / nbatches for loss in training_loss_terms]
        return training_loss, training_loss_terms

    def train_model(
        self,
        training: bool,
        train_stage: int,
        learning_rate: float,
    ) -> None:
        self.model = self.model_cls(
            self,
            {"training": training, "train_stage": train_stage},
        )
        lr = self.scheduler(
            learning_rate,
            {
                "lr_decay_steps": self.lr_decay_steps,
                "lr_decay_rate": self.lr_decay_rate,
                "weight_decay_rate": self.weight_decay_rate,
            },
        )
        lr_ini = learning_rate
        optimiser = self.optimiser(
            lr,
            {
                "lr_decay_steps": self.lr_decay_steps,
                "lr_decay_rate": self.lr_decay_rate,
                "weight_decay_rate": self.weight_decay_rate,
            },
        )
        loss = losses.apply_gradients(optimiser)

        ncol_loss = 4
        training_losses = np.zeros((self.epochs, ncol_loss))
        validation_losses = np.zeros((self.epochs, ncol_loss))
        testing_losses = np.zeros((self.epochs, ncol_loss))

        validation_loss_min = 1.0e9
        validation_iteration = 0
        validation_best_iteration = 0

        for epoch in self.tqdm(range(self.epochs)):
            is_best = False
            start_time = time.time()

            training_loss, training_loss_terms = self.train_step(
                epoch,
                loss,
            )

    @callback
    def train_colour(self) -> None:
        self.log.info("Training colour")
        self.epochs = self.epochs_colour
        self.train_model(training=True, train_stage=0, learning_rate=self.lr)

    @callback
    def train_latents(self) -> None:
        self.epochs = self.epochs_latent
        for latent in range(self.latent_dim):
            self.log.info(f"Training latent {latent + 1} + previous parameters")
            self.train_model(training=True, train_stage=latent, learning_rate=self.lr)

    @callback
    def train_amplitude(self) -> None:
        self.log.info("Training amplitude + previous parameters")
        self.epochs = self.epochs_amplitude
        self.train_model(
            training=True,
            train_stage=1 + self.latent_dim,
            learning_rate=self.lr,
        )

    @callback
    def train_all(self) -> None:
        self.log.info("Training all parameters")
        self.epochs = self.epochs_all
        self.train_model(
            training=True,
            train_stage=2 + self.latent_dim,
            learning_rate=self.lr_deltat,
        )

    @override
    def _is_completed(self) -> bool:
        return False

    @override
    def _load(self) -> None:
        return None

    @override
    def _run(self):
        for data in zip(
            self.train_data,
            self.test_data,
            self.validation_data,
            strict=True,
        ):
            self.training_data = data
            if self.split_training:
                # First train delta Av
                self.train_colour()
                # Then train latent parameters
                self.train_latents()
                # Then train delta M
                self.train_amplitude()
                # Then train delta T (and thus all parameters
                self.train_all()
            else:
                self.train_all()

        return True, None

    @override
    def _result(self):
        return True, None
