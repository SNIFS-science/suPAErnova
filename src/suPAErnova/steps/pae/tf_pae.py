import time
from typing import TYPE_CHECKING, Any, cast, final, override

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

from suPAErnova.steps import callback
from suPAErnova.steps.pae import PAEStep
from suPAErnova.config.pae.tf_pae import (
    prev,
    optional,
    required,
    optional_params,
    required_params,
)
from suPAErnova.model_utils.pae.tf_pae import (
    tf_losses as losses,
    tf_optimisers as optimisers,
    tf_schedulers as schedulers,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import typing as npt
    from tensorflow.python.types.core import PolymorphicFunction

    from suPAErnova.models.pae.tf_pae import TF_PAEModel
    from suPAErnova.utils.suPAErnova_types import CFG, CONFIG


@final
class TF_PAEStep(PAEStep):
    required = required
    optional = optional
    prev = prev
    required_params = required_params
    optional_params = optional_params

    def __init__(self, cfg: "CFG") -> None:
        super().__init__(cfg)

        # Model
        self.model_cls: type[TF_PAEModel]
        self.model: TF_PAEModel
        self.training_params: CFG
        self.encoder: ks.Model
        self.decoder: ks.Model
        self.checkpoint_ind: int
        self.checkpoint_layer: ks.layers.Layer

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
        self.sigma_time: float
        self.epoch: int
        self.training_stage: str

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
        self.current_kfold: int
        self.train_data: list[CFG]
        self.test_data: list[CFG]
        self.validation_data: list[CFG]
        self.training_data: tuple[
            CONFIG[tf.Tensor],
            CONFIG[tf.Tensor],
            CONFIG[tf.Tensor],
        ]
        self.validation_frac: float

    @staticmethod
    def dtype(dt: "npt.DTypeLike"):
        if np.issubdtype(dt, np.floating):
            return tf.float32
        if np.issubdtype(dt, np.integer):
            return tf.int32
        if np.issubdtype(dt, np.bool_):
            return tf.int32
        return tf.string

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
        self.training_stage = "NONE"

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
        self.current_kfold = -1
        self.train_data = [
            self.prep_data(self.data.train_data[i], "train") for i in self.kfold
        ]
        self.test_data = [self.data.test_data[i] for i in self.kfold]
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

        self.test_data = [
            self.prep_data(test_data, "test") for test_data in self.test_data
        ]
        self.validation_data = [
            self.prep_data(validation_data, "validation")
            for validation_data in self.validation_data
        ]

        return (True, None)

    def prep_data(self, data: "CFG", data_type: str) -> "CFG":
        redshift_mask = (data["redshift"] >= self.min_train_redshift) & (
            data["redshift"] <= self.max_train_redshift
        )
        phase_mask = (data["phase"] >= self.min_train_phase) & (
            data["phase"] <= self.max_train_phase
        )
        data[f"mask_{data_type}"] = redshift_mask & phase_mask
        return data
        # TODO: Add a `get_twins_mask` callback example

    def model_name(self, model: "TFAutoencoder", is_best: bool = False):
        dname = (
            self.outpath
            / f"AE__{self.latent_dim:02d}D__{'-'.join(str(e) for e in model.encode_dims)}-latent_layers__kfold{self.current_kfold:d}"
        )
        if not dname.is_dir():
            dname.mkdir(parents=True, exist_ok=True)

        fname = self.training_stage
        if is_best:
            fname += "__best"

        encoder_file = dname / f"{fname}__encoder.keras"
        decoder_file = dname / f"{fname}__decoder.keras"
        return encoder_file, decoder_file

    def save_model(
        self,
        model: "TFAutoencoder",
        batch: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        is_best: bool = False,
    ) -> None:
        if not model.save_model:
            return

        def _latent_means(
            mod: "TFAutoencoder",
        ) -> tuple[
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
        ]:
            zs = tf.map_fn(
                lambda x: mod.encode(*x),
                batch,
                fn_output_signature=tf.TensorSpec(
                    shape=(batch[0].shape[1], self.latent_dim + 3),
                ),
            )

            num_points = zs.shape[0] * zs.shape[1]
            mean_zs = zs / num_points

            mean_time = tf.reduce_sum(mean_zs[:, :, 0])
            mean_flux = tf.reduce_sum(mean_zs[:, :, 1])
            mean_colour = tf.reduce_sum(mean_zs[:, :, 2])

            return mean_time, mean_flux, mean_colour

        encoder_file, decoder_file = self.model_name(model, is_best)

        model_save = self.model_cls(self, {**model.training_params, "training": False})
        model_save.encoder.set_weights(model.encoder.get_weights())

        if model.physical_latent:
            # Calculate mean flux
            moving_means = _latent_means(model_save)

            # Create new model, normalising on mean flux
            model_save = self.model_cls(
                self,
                {
                    **model.training_params,
                    "training": False,
                    "moving_means": moving_means,
                },
            )
            model_save.encoder.set_weights(model.encoder.get_weights())

        model_save.decoder.set_weights(model.decoder.get_weights())

        model_save.encoder.save(encoder_file)
        model_save.decoder.save(decoder_file)

    def load_checkpoint(self) -> None:
        encoder_file, decoder_file = self.model_name(
            self.model,
            is_best=self.model.load_best,
        )
        self.encoder = ks.models.load_model(encoder_file, compile=False)
        self.decoder = ks.models.load_model(decoder_file, compile=False)

        if self.encoder is not None:
            self.checkpoint_ind, self.checkpoint_layer = next(
                (i, layer)
                for i, layer in reversed(list(enumerate(self.encoder.layers)))
                if isinstance(layer, ks.layers.Dense)
            )

    def train_step(self, loss: "PolymorphicFunction"):
        # Everything which is constant throughout each epoch
        train_data, test_data, validation_data = self.training_data
        train_flux = train_data["flux"]
        train_time = train_data["time"]
        train_sigma = train_data["sigma"]
        train_mask = train_data["mask"]
        train_shape = train_flux.shape
        mask_train = tf.cast(~tf.cast(train_data["mask_train"], tf.bool), tf.int32)

        test_flux = test_data["flux"]
        test_time = test_data["time"]
        test_sigma = test_data["sigma"]
        test_mask = test_data["mask"]
        mask_test = tf.cast(~tf.cast(test_data["mask_test"], tf.bool), tf.int32)

        validation_flux = validation_data["flux"]
        validation_time = validation_data["time"]
        validation_sigma = validation_data["sigma"]
        validation_mask = validation_data["mask"]
        mask_validation = tf.cast(
            ~tf.cast(validation_data["mask_validation"], tf.bool),
            tf.int32,
        )

        n_sn, _n_spectra, n_flux = train_shape
        n_batches = n_sn // self.batch_size

        dflux = tf.zeros(train_shape)
        if self.mask_frac > 0:
            dmask = tf.ones(train_shape, dtype=tf.int32)
            # Get the number of unmasked spectra per SN
            # (n_sn, n_flux)
            n_spectra_to_shuffle_per_SN = tf.reduce_sum(
                train_mask,
                axis=1,
            )

            # Calculate the number of spectra to mask for each SN
            # (n_sn, n_flux)
            n_spectra_to_mask_per_SN = tf.cast(
                self.mask_frac * tf.cast(n_spectra_to_shuffle_per_SN, tf.float32),
                tf.int32,
            )

            # The indicies of unmasked spectra
            # (n_unmasked_spectra, 3)
            unmasked_indices = tf.cast(tf.where(train_mask), dtype=tf.int32)

            mask_sn_indices = unmasked_indices[:, 0]  # SN indices
            mask_spectra_indices = unmasked_indices[:, 1]  # Spectra indices

            # Gather the number of spectra to mask for each SN
            n_spectra_to_mask = tf.gather(n_spectra_to_mask_per_SN, mask_sn_indices)
            mask_flags = mask_spectra_indices <= n_spectra_to_mask[:, 0]

            # (n_spectra_to_mask, 3)
            mask_indices = tf.boolean_mask(
                unmasked_indices,
                mask_flags,
            )

            mask_update = tf.zeros(
                [
                    tf.shape(mask_indices)[0],
                ],  # Since the shape of mask_indices is dynamic, use tf.shape
                dtype=tf.int32,
            )

            # Mask fraction of spectra
            dmask = tf.tensor_scatter_nd_update(
                dmask,
                mask_indices,
                mask_update,
            )

            n_spectra_to_shuffle = tf.gather(
                n_spectra_to_shuffle_per_SN,
                mask_sn_indices,
            )

            shuffle_flags = mask_spectra_indices <= n_spectra_to_shuffle[:, 0]
            shuffle_indices = tf.boolean_mask(unmasked_indices, shuffle_flags)[
                :,
                :-1,
            ][::n_flux]

            shuffle_sn_indices = shuffle_indices[:, 0]  # SN indices
            shuffle_spectra_indices = shuffle_indices[:, 1]  # Spectra indices

            ragged_tensor = tf.RaggedTensor.from_value_rowids(
                shuffle_spectra_indices,
                shuffle_sn_indices,
            )
        else:
            dmask = mask_train

        # # Vary phase by observational uncertainty
        train_dphase = train_data["dphase"]
        dtime = tf.zeros([n_sn, 1, 1])

        # Everything which varies between each epoch
        @tf.function
        def _train_step(
            dflux_orig,
            dmask_orig,
            dtime_orig,
        ) -> tuple[
            tf.Tensor,
            tf.Tensor,
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ]:
            tf.random.set_seed(self.epoch)

            # # Add noise during training drawn from observational uncertainty
            if self.noise_scale > 0:
                dflux = self.noise_scale * tf.abs(
                    train_sigma * tf.random.normal(train_shape),
                )
            else:
                dflux = dflux_orig

            # Mask a random fraction of spectra
            if self.mask_frac > 0:
                shuffled_indices = tf.ragged.map_flat_values(
                    tf.random.shuffle,
                    ragged_tensor,
                )

                # Get number of selected elements in total
                num_selected = tf.shape(shuffled_indices.flat_values)[0]

                # Get batch indices using row_lengths() and tf.range()
                batch_indices = tf.repeat(
                    tf.range(shuffled_indices.nrows()),
                    shuffled_indices.row_lengths(),
                )

                # Stack to create indices for gather_nd
                gather_nd_indices = tf.stack(
                    [batch_indices, shuffled_indices.flat_values],
                    axis=1,
                )

                # Gather from dmask
                shuffle_update = tf.gather_nd(
                    dmask_orig,
                    gather_nd_indices,
                )

                # Shuffle masked spectra with unmasked
                dmask = tf.tensor_scatter_nd_update(
                    dmask_orig,
                    shuffle_indices,
                    shuffle_update,
                )

                dmask *= mask_train
            else:
                dmask = dmask_orig

            if self.sigma_time == 0:
                dtime = (train_dphase / 50) * tf.random.normal([
                    n_sn,
                    1,
                    1,
                ])
            elif self.sigma_time > 0:
                dtime = (self.sigma_time / 50) * tf.random.normal([n_sn, 1, 1])
            else:
                dtime = dtime_orig

            # shuffle indices each epoch for batches
            # batch feeding can be improved, but the various types of specialized masks/dropout
            # are easy to implement in this non-optimized fashion, and the dataset is small
            mask = train_mask * dmask
            inds = tf.random.shuffle(tf.range(n_sn) % n_batches)
            batch_mask = tf.stack(tf.dynamic_partition(mask, inds, n_batches))

            # Repeatedly shuffle until all batches have at least one valid SN
            valid_batches = tf.reduce_all([
                tf.reduce_sum(batch_mask[i]) for i in range(n_batches)
            ])
            while not valid_batches:
                inds = tf.random.shuffle(tf.range(n_sn) % n_batches)
                batch_mask = tf.stack(tf.dynamic_partition(mask, inds, n_batches))
                valid_batches = tf.reduce_all([
                    tf.reduce_sum(batch_mask[i]) for i in range(n_batches)
                ])

            flux = train_flux + dflux
            batch_flux = tf.stack(tf.dynamic_partition(flux, inds, n_batches))

            time = train_time + dtime
            batch_time = tf.stack(tf.dynamic_partition(time, inds, n_batches))

            sigma = train_sigma
            batch_sigma = tf.stack(tf.dynamic_partition(sigma, inds, n_batches))

            batch = (batch_flux, batch_time, batch_mask, batch_sigma)

            batch_results = tf.map_fn(
                lambda x: loss(*x),
                batch,
                fn_output_signature=tf.TensorSpec(shape=(3,)),
            )

            # Overall training loss is the sum of the loss of each batch
            training_loss = tf.reduce_sum(batch_results[:, 0])
            training_loss_terms = tf.reduce_sum(batch_results, axis=0) / n_batches

            return training_loss, training_loss_terms, batch

        @tf.function
        def _validation_step():
            validation_results = loss(
                validation_flux,
                validation_time,
                validation_mask * mask_validation,
                validation_sigma,
            )
            validation_loss = validation_results[0]
            return validation_loss, validation_results

        @tf.function
        def _test_step():
            test_results = loss(
                test_flux,
                test_time,
                test_mask * mask_test,
                test_sigma,
            )
            test_loss = test_results[0]
            return test_loss, test_results

        return (
            lambda: _train_step(dflux, dmask, dtime),
            _validation_step,
            _test_step,
        )

    # TODO: Make tf.function
    def train_model(self) -> None:
        lr_ini = self.model.learning_rate
        lr = self.scheduler(
            lr_ini,
            {
                "lr_decay_steps": self.lr_decay_steps,
                "lr_decay_rate": self.lr_decay_rate,
                "weight_decay_rate": self.weight_decay_rate,
            },
        )
        optimiser = self.optimiser(
            lr,
            {
                "lr_decay_steps": self.lr_decay_steps,
                "lr_decay_rate": self.lr_decay_rate,
                "weight_decay_rate": self.weight_decay_rate,
            },
        )
        loss = losses.apply_gradients(optimiser, self.model)
        train_step, validation_step, test_step = self.train_step(loss)

        ncol_loss = 4
        train_losses = np.zeros((self.epochs, ncol_loss))
        validation_losses = np.zeros((self.epochs, ncol_loss))
        test_losses = np.zeros((self.epochs, ncol_loss))

        validation_loss_min = 1.0e9
        iteration = 0
        best_iteration = 0

        for epoch in self.tqdm(range(self.epochs)):
            self.epoch = epoch
            is_best = False
            start_time = time.time()
            _train_loss, train_loss_terms, batch = train_step()

            # Get average loss over batches
            train_losses[epoch, 0] = epoch
            train_losses[epoch, 1:] = train_loss_terms

            end_time = time.time()

            if epoch % self.validate_every_n == 0:
                t_epoch = end_time - start_time

                validation_loss, validation_loss_terms = validation_step()
                validation_losses[iteration, 0] = epoch
                validation_losses[iteration, 1:] = validation_loss_terms

                _test_loss, test_loss_terms = test_step()
                test_losses[iteration, 0] = epoch
                test_losses[iteration, 1:] = test_loss_terms

                self.log.debug(
                    # f"\nepoch={epoch:d}, time={end_time - start_time:.3f}s\n (total_loss, loss_recon, loss_cov)\ntrain loss: {train_loss_terms[0]:.2E} {train_loss_terms[1]:.2E} {train_loss_terms[2]:.2E}\nval loss: {validation_loss_terms[0]:.2E} {validation_loss_terms[1]:.2E} {validation_loss_terms[2]:.2E}\ntest loss: {test_loss_terms[0]:.2E} {test_loss_terms[1]:.2E} {test_loss_terms[2]:.2E}",
                    f"\nepoch={epoch:d}, time={end_time - start_time:.3f}s\n (total_loss, loss_recon, loss_cov)\ntrain loss: {train_loss_terms[0]:.2E}\nval loss: {validation_loss_terms[0]:.2E}\ntest loss: {test_loss_terms[0]:.2E}",
                )

                previous_validation_decrease = iteration - best_iteration
                if validation_loss < validation_loss_min:
                    self.log.debug("Best validation epoch so far, saving model")
                    best_iteration = iteration
                    validation_loss_min = min(validation_loss_min, validation_loss)
                    self.save_model(
                        self.model,
                        batch[:-1],
                        is_best=True,
                    )  # Drop batch_sigma
                iteration += 1
        self.save_model(self.model, batch[:-1])  # Drop batch_sigma
        self.load_checkpoint()  # Load best (or last) checkpoint in preperation for the next training stage

    @callback
    def train_colour(self) -> None:
        self.log.info("Training colour")
        self.epochs = self.epochs_colour
        self.training_stage = "colour"
        self.training_params = {
            "training": True,
            "train_stage": 0,
            "learning_rate": self.lr,
        }
        self.model = self.model_cls(self, self.training_params)
        self.train_model()

    @callback
    def train_latents(self) -> None:
        self.epochs = self.epochs_latent
        for latent in range(self.latent_dim):
            self.log.info(f"Training latent {latent + 1} + previous parameters")
            self.training_stage = f"latent_{latent + 1}"
            self.training_params = {
                "training": True,
                "train_stage": latent + 1,
                "learning_rate": self.lr,
            }
            self.model = self.model_cls(self, self.training_params)
            init_weights = self.model.encoder.layers[self.checkpoint_ind].get_weights()[
                0
            ]
            weights = self.checkpoint_layer.get_weights()[0]
            # Overwrite previous checkpoint's training
            weights[:, latent + 3] = init_weights[:, latent + 3] / 100  # TODO: Why 100?
            self.encoder.layers[self.checkpoint_ind].set_weights([weights])
            self.model.encoder.set_weights(self.encoder.get_weights())
            self.model.decoder.set_weights(self.decoder.get_weights())
            self.train_model()

    @callback
    def train_amplitude(self) -> None:
        self.log.info("Training amplitude + previous parameters")
        self.epochs = self.epochs_amplitude
        self.training_stage = "amplitude"
        self.training_params = {
            "training": True,
            "train_stage": self.latent_dim + 1,
            "learning_rate": self.lr,
        }
        self.model = self.model_cls(self, self.training_params)
        init_weights = self.model.encoder.layers[self.checkpoint_ind].get_weights()[0]
        weights = self.checkpoint_layer.get_weights()[0]
        # Overwrite previous checkpoint's training
        weights[:, 1] = init_weights[:, 1] / 100  # TODO: Why 100?
        self.encoder.layers[self.checkpoint_ind].set_weights([weights])
        self.model.encoder.set_weights(self.encoder.get_weights())
        self.model.decoder.set_weights(self.decoder.get_weights())
        self.train_model()

    @callback
    def train_time(self) -> None:
        self.log.info("Training time + previous parameters")
        self.epochs = self.epochs_all
        self.training_stage = "all"
        self.training_params = {
            "training": True,
            "train_stage": self.latent_dim + 2,
            "learning_rate": self.lr_deltat,
        }
        self.model = self.model_cls(self, self.training_params)
        init_weights = self.model.encoder.layers[self.checkpoint_ind].get_weights()[0]
        weights = self.checkpoint_layer.get_weights()[0]
        # Overwrite previous checkpoint's training
        weights[:, 0] = init_weights[:, 0] / 100  # TODO: Why 100?
        self.train_model()

    @callback
    def train_all(self) -> None:
        self.log.info("Training all parameters")
        self.epochs = self.epochs_all
        self.training_stage = "all"
        self.training_params = {
            "training": True,
            "train_stage": self.latent_dim + 2,
            "learning_rate": self.lr_deltat,
        }
        self.model = self.model_cls(self, self.training_params)
        self.train_model()

    @override
    def _is_completed(self) -> bool:
        return False

    @override
    def _load(self) -> None:
        return None

    @override
    def _run(self):
        # tf.debugging.enable_check_numerics()
        # tf.debugging.enable_traceback_filtering()
        for kfold, data in enumerate(
            zip(
                self.train_data,
                self.test_data,
                self.validation_data,
                strict=True,
            ),
        ):
            self.current_kfold = kfold
            self.training_data = cast(
                "tuple[CONFIG[tf.Tensor], CONFIG[tf.Tensor], CONFIG[tf.Tensor]]",
                tuple(
                    {
                        k: tf.convert_to_tensor(v, dtype=self.dtype(v.dtype))
                        for k, v in d.items()
                    }
                    for d in data
                ),
            )

            if self.split_training:
                # First train delta Av
                self.train_colour()
                # Then train latent parameters
                self.train_latents()
                # Then train delta M
                self.train_amplitude()
                # Then train delta T (and thus all parameters)
                self.train_time()
            else:
                self.train_all()

            # Clear GPU Memory between kfolds
            ks.backend.clear_session()

        return True, None

    @override
    def _result(self):
        return True, None
