# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    NamedTuple,
    cast,
    override,
)

import keras
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.pae.tf import (
        S,
        FTensor,
        ITensor,
        TFPAEModel,
        TensorCompatible,
    )
    from suPAErnova.steps.nflow.tf import TFNFlowModel
    from suPAErnova.steps.posterior.model import PosteriorModel
    from suPAErnova.configs.steps.posterior.tf import TFPosteriorModelConfig

    from .model import Stage

    # === Custom Types ===

    PosteriorData = tuple[
        FTensor[S["batch_dim nspec_dim phase_dim"]],
        FTensor[S["batch_dim nspec_dim wl_dim"]],
        FTensor[S["batch_dim nspec_dim wl_dim"]],
    ]
    PosteriorMask = FTensor[S["batch_dim nspec_dim wl_dim"]]
    PosteriorVars = tuple[
        FTensor[S["batch_dim nspec_dim n_μs"]],
        FTensor[S["batch_dim nspec_dim 1"]],
        FTensor[S["batch_dim nspec_dim 1"]],
        FTensor[S["batch_dim nspec_dim 1"]],
    ]
    PosteriorInputs = tuple[PosteriorData, PosteriorVars]
    PosteriorOutputs = FTensor[S["batch_dim nspec_dim n_μs"]]


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPosteriorLoss(ks.losses.Loss):
    @override
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return -y_pred


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

        # --- Previous Steps ---
        self.nflow: "TFNFlowModel" = cast("TFNFlowModel", config.nflow.model)

        # --- Variables ---
        # Initial values
        self.initial_bias: "FTensor[S['batch_dim nspec_dim 1']]"
        self.initial_delta_m: "FTensor[S['batch_dim nspec_dim 1']]"
        self.initial_delta_p: "FTensor[S['batch_dim nspec_dim 1']]"
        self.initial_us: "FTensor[S['batch_dim nspec_dim n_μs']]"
        self.initial_latents: "FTensor[S['batch_dim nspec_dim n_latents']]"

        # Updated values
        self.bias: tf.Variable
        self.delta_m: tf.Variable
        self.delta_p: tf.Variable
        self.us: "FTensor[S['batch_dim nspec_dim n_μs']]"
        self.latents: "FTensor[S['batch_dim nspec_dim n_latents']]"
        self.data: "PosteriorData"
        self.pos: "PosteriorVars"
        self.mask: "PosteriorMask"
        self.recon_error: tf.Tensor
        self.recon_error_centers: tf.Tensor

        # --- Training ---
        self.built: bool = False
        self._chain: int = 0
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"
        self.n_samples: int
        self.nspec_dim: int
        self.random_initial_positions: bool = options.random_initial_positions
        self.tolerance: float = options.tolerance
        self.max_iterations: int = options.max_iterations

        self._loss: ks.losses.Loss = (
            options.loss_cls() if options.loss_cls is not None else TFPosteriorLoss()
        )
        self.stage: Stage

        # --- Layers ---
        self.n_us: int = self.nflow.pae.n_latents - 2  # ΔAᵥ + zs
        self.u_prior: tfd.MultivariateNormalDiag = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.n_us), scale_diag=tf.ones(self.n_us)
        )

        self.train_delta_m: bool = options.train_delta_m
        self.delta_m_mean: float = options.delta_m_mean
        self.delta_m_std: float = options.delta_m_std
        self.delta_m_prior: tdf.Normal = tfd.Normal(
            loc=self.delta_m_mean, scale=self.delta_m_std
        )

        self.train_delta_p: bool = options.train_delta_p
        self.delta_p_mean: float = options.delta_p_mean
        self.delta_p_std: float = options.delta_p_std
        self.delta_p_prior: tfd.Normal = tfd.Normal(
            loc=self.delta_p_mean, scale=self.delta_p_std
        )

        self.u_likelihood: tfd.Independent

        self.results: NamedTuple

        # --- Metrics ---
        self._loss_terms: dict[str, "FTensor[S['']]"]
        self.chain_min: tf.Tensor
        self.converged: tf.Tensor
        self.num_evals: tf.Tensor
        self.neg_log_like: tf.Tensor
        self.delta_m_val: tf.Tensor
        self.init_delta_m_val: tf.Tensor
        self.delta_p_val: tf.Tensor
        self.init_delta_p_val: tf.Tensor
        self.us_val: tf.Tensor
        self.init_us_val: tf.Tensor
        self.bias_val: tf.Tensor
        self.init_bias_val: tf.Tensor
        self.initialised: bool = False
        self.loss_tracker: ks.metrics.Metric = ks.metrics.Mean("loss")

    @property
    @override
    def metrics(self) -> list[ks.metrics.Metric]:
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each chain
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker
            # self.chain_min,
            # self.converged,
            # self.num_evals,
            # self.neg_log_like,
            # self.delta_m_val,
            # self.init_delta_m_val,
            # self.delta_p_val,
            # self.init_delta_p_val,
            # self.us_val,
            # self.init_us_val,
        ]

    @override
    def call(
        self,
        inputs: "PosteriorInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "PosteriorOutputs":
        training = False if training is None else training
        ((input_phase, input_amplitude, input_d_amplitude), pos) = inputs

        pos = cast("FTensor[S['batch_dim nspec_dim n_μs + 3']]", cast("object", pos))
        us = pos[:, :, : self.n_us]
        delta_m = pos[:, :, self.n_us : self.n_us + 1]
        delta_p = pos[:, :, self.n_us + 1 : self.n_us + 2]
        bias = pos[:, :, self.n_us + 2 :]
        zs = tf.concat(
            (
                self.nflow.pdf.bijector.forward(us),
                delta_m,
                delta_p,
            ),
            axis=-1,
        )
        zs = ks.layers.RepeatVector(self.nspec_dim)(zs[:, 0, :])

        input_mask = tf.cast(
            (
                mask
                if mask is not None
                else tf.ones_like(input_amplitude, dtype=tf.int32)
            ),
            tf.float32,
        )

        amplitude = self.nflow.pae.decoder(
            (zs, input_phase), training=False, mask=input_mask
        )

        if not self.nflow.physical_latents:
            amplitude *= delta_m
        amplitude += bias

        # Measured average AE reconstruction error at current times
        mean_pae_recon_error = tf.transpose(
            tfp.math.interp_regular_1d_grid(
                x=tf.transpose(input_phase + delta_p),
                x_ref_min=self.recon_error_centers[0],
                x_ref_max=self.recon_error_centers[-1],
                y_ref=self.recon_error,
            )
        )[:, 0, :]

        std_amplitude = tf.sqrt(
            (amplitude * mean_pae_recon_error) ** 2 + input_d_amplitude**2
        )
        std_amplitude *= input_mask
        std_amplitude += 1 - input_mask

        self.u_likelihood = tfd.Independent(
            tfd.MultivariateNormalDiag(loc=amplitude, scale_diag=std_amplitude),
            reinterpreted_batch_ndims=0,
        )

        log_prior: "FTensor[S['batch_dim nspec_dim']]" = self.u_prior.log_prob(us)

        log_likelihood: "FTensor[S['batch_dim']]" = self.u_likelihood.log_prob(
            input_amplitude * input_mask
        )

        mean_log_likelihood = tf.reduce_sum(
            log_likelihood, axis=-1, keepdims=True
        ) / tf.reduce_sum(input_mask)

        return log_prior + mean_log_likelihood

    @override
    def __call__(
        self,
        inputs: "PosteriorInputs",
        *,
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "FTensor[S['batch_dim nspec_dim n_μs']]":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)

    @override
    def train_step(
        self,
        data: "TensorCompatible",
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:
        training = True

        # === Per Chain Setup ===
        # Fixed RNG
        tf.random.set_seed(self._chain)
        self._chain += 1

        self.prep_data()
        phase, amplitude, d_amplitude = self.data
        mask = self.mask
        position = self.pos

        position = tf.concat(position, axis=-1)
        init_us = position[:, :, : self.n_us]
        init_delta_m = position[:, :, self.n_us : self.n_us + 1]
        init_delta_p = position[:, :, self.n_us + 1 : self.n_us + 2]
        init_bias = position[:, :, self.n_us + 2 :]

        def vals_and_grads(pos: "PosteriorVars"):
            inputs = ((phase, amplitude, d_amplitude), pos)
            log_prob = self(inputs, training=True, mask=mask)
            return self._loss(tf.zeros_like(log_prob), log_prob)

        def lbfgs(x: "PosteriorVars"):
            return tfp.math.value_and_gradient(vals_and_grads, x)

        self.results = tfp.optimizer.lbfgs_minimize(
            lbfgs,
            initial_position=position,
            tolerance=self.tolerance,
            x_tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            num_correction_pairs=1,
        )

        pos = self.results.position
        us = pos[:, :, : self.n_us]
        delta_m = pos[:, :, self.n_us : self.n_us + 1]
        delta_p = pos[:, :, self.n_us + 1 : self.n_us + 2]
        bias = pos[:, :, self.n_us + 2 :]

        self.pos = (
            tf.convert_to_tensor(self.us),
            tf.convert_to_tensor(self.delta_m),
            tf.convert_to_tensor(self.delta_p),
            tf.convert_to_tensor(self.bias),
        )

        if not self.initialised:
            self.chain_min = tf.zeros(self.n_samples)
            self.converged = self.results.converged
            self.num_evals = self.results.num_objective_evaluations
            self.neg_log_like = self.results.objective_value
            self.delta_m_val = delta_m
            self.init_delta_m_val = init_delta_m
            self.delta_p_val = delta_p
            self.init_delta_p_val = init_delta_p
            self.us_val = us
            self.init_us_val = init_us
            self.bias_val = bias
            self.init_bias_val = init_bias
            self.initialised = True
        else:
            improved = self.results.objective_value < self.neg_log_like
            self.chain_min = tf.where(
                improved[:, 0],
                self._chain * tf.ones_like(self.chain_min),
                self.chain_min,
            )
            self.converged = tf.where(improved, self.results.converged, self.converged)
            self.num_evals += self.results.num_objective_evaluations
            self.neg_log_like = tf.where(
                improved, self.results.objective_value, self.neg_log_like
            )
            self.delta_m_val = tf.where(
                tf.expand_dims(improved, axis=-1), delta_m, self.delta_m_val
            )
            self.init_delta_m_val = tf.where(
                tf.expand_dims(improved, axis=-1), init_delta_m, self.init_delta_m_val
            )
            self.delta_p_val = tf.where(
                tf.expand_dims(improved, axis=-1), delta_p, self.delta_p_val
            )
            self.init_delta_p_val = tf.where(
                tf.expand_dims(improved, axis=-1), init_delta_p, self.init_delta_p_val
            )
            self.us_val = tf.where(tf.expand_dims(improved, axis=-1), us, self.us_val)
            self.init_us_val = tf.where(
                tf.expand_dims(improved, axis=-1), init_us, self.init_us_val
            )
            self.bias_val = tf.where(
                tf.expand_dims(improved, axis=-1), bias, self.bias_val
            )
            self.init_bias_val = tf.where(
                tf.expand_dims(improved, axis=-1), init_bias, self.init_bias_val
            )

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train_model(
        self,
        stage: "Stage",
    ) -> ks.callbacks.History:
        self.stage = stage

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        if self.stage.savepath is not None:
            # --- Backup & Restore ---
            # Backup checkpoints each chain and restore if training got cancelled midway through
            if not self.force:
                backup_dir = self.stage.savepath / "backups"
                backup_callback = ks.callbacks.BackupAndRestore(str(backup_dir))
                callbacks.append(backup_callback)

            # --- Model Checkpoint ---
            checkpoint_callback = ks.callbacks.ModelCheckpoint(
                str(self.stage.savepath / self.weights_path),
                save_best_only=self.save_best,
                save_weights_only=True,
                # verbose=0,
            )
            callbacks.append(checkpoint_callback)

        # --- Terminate on NaN ---
        # Terminate training when a NaN loss is encountered
        callbacks.append(ks.callbacks.TerminateOnNaN())

        self.prep_data()

        if stage.loadpath is not None:
            self.load_checkpoint(stage.loadpath)
        else:
            self.build_model()

        # === Train ===
        self._chain = 0
        return self.fit(
            x=(*self.data, self.mask),
            y=self.pos,
            initial_epoch=self._chain,
            epochs=self.stage.n_chains,
            batch_size=self.mask.shape[0],
            callbacks=callbacks,
            # verbose=0,
        )

    def build_model(
        self,
    ) -> None:
        if not self.built:
            loss = self._loss
            self.compile(
                loss=loss,
                run_eagerly=self.stage.debug,
            )
            self(
                (self.data, tf.concat(self.pos, axis=-1)),
                training=False,
                mask=self.mask,
            )

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

    def prep_data(self) -> None:
        # === Prep Data ===
        train_phase = self.stage.train_data.phase
        train_d_phase = self.stage.train_data.dphase
        train_amplitude = self.stage.train_data.amplitude
        train_d_amplitude = self.stage.train_data.sigma
        train_mask = self.stage.train_data.mask
        self.n_samples = train_phase.shape[0]
        self.nspec_dim = train_phase.shape[1]

        recon_error, _recon_error_edges, recon_error_centers = (
            self.nflow.pae.recon_error((
                tf.convert_to_tensor(train_phase, dtype=tf.float32),
                tf.convert_to_tensor(train_d_phase, dtype=tf.float32),
                tf.convert_to_tensor(train_amplitude, dtype=tf.float32),
                tf.convert_to_tensor(train_d_amplitude, dtype=tf.float32),
                tf.convert_to_tensor(train_mask, dtype=tf.int32),
            ))
        )
        self.recon_error = recon_error
        self.recon_error_centers = recon_error_centers

        initial_bias = tf.zeros((self.n_samples, 1))
        self.initial_bias = ks.layers.RepeatVector(self.nspec_dim)(initial_bias)
        self.bias = tf.Variable(self.initial_bias)
        self.bias.assign(self.initial_bias)
        # TODO: This is terrible, make it better
        if self.stage.name == "init":
            if self.random_initial_positions:
                initial_delta_m = tf.expand_dims(
                    self.delta_m_prior.sample(self.n_samples), axis=-1
                )
                self.initial_delta_m = ks.layers.RepeatVector(self.nspec_dim)(
                    initial_delta_m
                )
                initial_delta_p = tf.expand_dims(
                    self.delta_p_prior.sample(self.n_samples), axis=-1
                )
                self.initial_delta_p = ks.layers.RepeatVector(self.nspec_dim)(
                    initial_delta_p
                )
                initial_us = self.u_prior.sample(self.n_samples)
                self.initial_us = ks.layers.RepeatVector(self.nspec_dim)(initial_us)
                initial_latents = self.nflow.pdf.bijector.forward(initial_us)
                self.initial_latents = ks.layers.RepeatVector(self.nspec_dim)(
                    initial_latents
                )
            else:
                self.initial_latents = self.nflow.pae.encoder(
                    (
                        tf.convert_to_tensor(train_phase),
                        tf.convert_to_tensor(train_amplitude),
                    ),
                    training=False,
                    mask=train_mask,
                )
                initial_us = self.nflow.pdf.bijector.inverse(
                    self.nflow.get_data(self.initial_latents)
                )
                self.initial_us = ks.layers.RepeatVector(self.nspec_dim)(initial_us)
                self.initial_delta_m = self.initial_latents[:, :, -2:-1]
                self.initial_delta_p = self.initial_latents[:, :, -1:]
        elif self.stage.name == "early":
            initial_delta_m = tf.expand_dims(
                self.delta_m_prior.sample(self.n_samples), axis=-1
            )
            self.initial_delta_m = ks.layers.RepeatVector(self.nspec_dim)(
                initial_delta_m
            )
            initial_delta_p = tf.expand_dims(
                self.delta_p_prior.sample(self.n_samples), axis=-1
            )
            self.initial_delta_p = ks.layers.RepeatVector(self.nspec_dim)(
                initial_delta_p
            )
            initial_us = self.u_prior.sample(self.n_samples)
            self.initial_us = ks.layers.RepeatVector(self.nspec_dim)(initial_us)
            initial_latents = self.nflow.pdf.bijector.forward(initial_us)
            self.initial_latents = ks.layers.RepeatVector(self.nspec_dim)(
                initial_latents
            )
        elif self.stage.name == "mid":
            delta_m_max = 1.5
            delta_m_min = -1.5
            d_delta_m = (delta_m_max - delta_m_min) / (10 - 1)
            initial_delta_m = tf.expand_dims(
                tf.zeros(self.n_samples) + delta_m_min + (self._chain - 10) * d_delta_m,
                axis=-1,
            )
            self.initial_delta_m = ks.layers.RepeatVector(self.nspec_dim)(
                initial_delta_m
            )
            initial_delta_p = tf.expand_dims(
                self.delta_p_prior.sample(self.n_samples), axis=-1
            )
            self.initial_delta_p = ks.layers.RepeatVector(self.nspec_dim)(
                initial_delta_p
            )
            initial_us = self.u_prior.sample(self.n_samples) * 0.0
            self.initial_us = ks.layers.RepeatVector(self.nspec_dim)(initial_us)
            initial_latents = self.nflow.pdf.bijector.forward(initial_us)
            self.initial_latents = ks.layers.RepeatVector(self.nspec_dim)(
                initial_latents
            )
        else:
            pass
            # initial_us = self.u_prior.sample(self.n_samples) * 0.0
            # initial_latents = self.nflow.pdf.bijector.forward(initial_us)
            # delta_av_max = 0.5
            # delta_av_min = -0.5
            # d_delta_av = (delta_av_max - delta_av_min) / self.stage.n_chains
            # delta_av = (
            #     tf.zeros(self.n_samples) + delta_av_min + self._chain * d_delta_av
            # )
            # initial_latents[:, 0] = delta_av
            # initial_us = self.nflow.pdf.bijector.inverse(initial_latents)
            # self.initial_us = ks.layers.RepeatVector(self.nspec_dim)(initial_us)
            # self.initial_latents = ks.layers.RepeatVector(self.nspec_dim)(
            #     initial_latents
            # )
            #
            # initial_delta_m = tf.expand_dims(
            #     self.delta_m_prior.sample(self.n_samples) * 0.0, axis=-1
            # )
            # self.initial_delta_m = ks.layers.RepeatVector(self.nspec_dim)(
            #     initial_delta_m
            # )
            # initial_delta_p = tf.expand_dims(
            #     self.delta_p_prior.sample(self.n_samples), axis=-1
            # )
            # self.initial_delta_p = ks.layers.RepeatVector(self.nspec_dim)(
            #     initial_delta_p
            # )

        self.delta_m = tf.Variable(self.initial_delta_m)
        self.delta_m.assign(self.initial_delta_m)

        self.delta_p = tf.Variable(self.initial_delta_p)
        self.delta_p.assign(self.initial_delta_p)

        self.us = self.initial_us

        self.latents = self.initial_latents

        self.data = (
            tf.convert_to_tensor(train_phase, dtype=tf.float32),
            tf.convert_to_tensor(train_amplitude, dtype=tf.float32),
            tf.convert_to_tensor(train_d_amplitude, dtype=tf.float32),
        )
        self.mask = tf.convert_to_tensor(train_mask, dtype=tf.int32)
        self.pos = (
            tf.convert_to_tensor(self.us),
            tf.convert_to_tensor(self.delta_m),
            tf.convert_to_tensor(self.delta_p),
            tf.convert_to_tensor(self.bias),
        )
