# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, Any, Literal, cast, final, override
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

if TYPE_CHECKING:
    from logging import Logger
    from collections.abc import Callable

    from suPAErnova.steps.data.data import SNPAEData
    from suPAErnova.steps.pae.model import Stage, PAEModel
    from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig

    AmplitudeInput = tf.Tensor
    PhaseInput = tf.Tensor
    MaskInput = tf.Tensor

    DeltaAvLatent = tf.Tensor
    DeltaMLatent = tf.Tensor
    DeltaPLatent = tf.Tensor
    ZsLatent = tf.Tensor

    EncodeInputs = tuple[AmplitudeInput, PhaseInput, MaskInput]
    EncodeOutputs = tf.Tensor

    DecodeInputs = tuple[EncodeOutputs, PhaseInput, MaskInput]
    DecodeOutputs = tf.Tensor

    from tensorflow._aliases import TensorCompatible


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEEncoder(ks.layers.Layer):
    def __init__(
        self,
        config: "PAEModel",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{config.name.split()[-1]}Encoder", **kwargs)

        options = cast("TFPAEModelConfig", config.options)

        # --- Config Params ---
        self.architecture: Literal["dense", "convolutional"] = options.architecture

        self.nspec_dim: int = int(config.nspec_dim)
        self.wl_dim: int = int(config.wl_dim)
        self.phase_dim: int = int(config.phase_dim)

        self.n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents
        self.n_latents: int = self.n_physical + self.n_zs

        self.encode_dims: list[int] = options.encode_dims
        self.n_encode_dims: int = len(self.encode_dims)

        self.activation_fn: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn
        self.kernel_regulariser_cls: ks.regularizers.Regularizer = (
            options.kernel_regulariser_cls(options.kernel_regulariser_penalty)
        )

        self.dropout: float = options.dropout
        self.batch_normalisation: bool = options.batch_normalisation

        # --- Training Params ---
        self.training: bool
        self.stage_mask: tf.Tensor

        # --- Tensor Specs ---
        self.input_amp_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, self.wl_dim), dtype=tf.float32, name="amplitude"
        )
        self.input_phase_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, self.phase_dim), dtype=tf.float32, name="phase"
        )
        self.input_mask_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, self.wl_dim), dtype=tf.int32, name="mask"
        )
        self.output_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, self.n_latents), dtype=tf.float32, name="outputs"
        )

        # --- Layers ---
        self.encode_layers: list[ks.layers.Dense | ks.layers.Conv2D]
        self.dropout_layers: list[ks.layers.Dropout | ks.layers.Identity]
        self.batch_normalisation_layers: list[
            ks.layers.BatchNormalization | ks.layers.Identity
        ]
        self.encode_nspec_layer: ks.layers.Dense
        self.encode_output_layer: ks.layers.Dense

    @override
    @tf.function
    def build(self, input_shape) -> None:
        # Encode from input layer dimensions into intermediate dimensions
        self.encode_layers = []
        self.dropout_layers = []
        self.batch_normalisation_layers = []
        for i in range(self.n_encode_dims):
            n = self.encode_dims[i]
            if self.architecture == "dense":
                self.encode_layers.append(
                    ks.layers.Dense(
                        n,
                        activation=self.activation_fn,
                        kernel_regularizer=self.kernel_regulariser_cls,
                    )
                )
            else:
                self.encode_layers.append(
                    ks.layers.Conv2D(
                        n,
                        kernel_size=(1, self.kernel_size),
                        strides=(1, self.stride),
                        activation=self.activation_fn,
                        kernel_regularizer=self.kernel_regulariser_cls,
                        padding="same",
                    )
                )

            # Dropout layer
            if self.dropout > 0:
                self.dropout_layers.append(
                    ks.layers.Dropout(self.dropout, noise_shape=[None, 1, None])
                )
            else:
                self.dropout_layers.append(ks.layers.Identity())

            # Batch normalisation layer
            if self.batch_normalisation:
                self.batch_normalisation_layers.append(ks.layers.BatchNormalization())
            else:
                self.batch_normalisation_layers.append(ks.layers.Identity())

        # Encode from intermediate dimensions into nspec_dim dimensions
        self.encode_nspec_layer = ks.layers.Dense(
            self.nspec_dim,
            activation=self.activation_fn,
            kernel_regularizer=self.kernel_regulariser_cls,
        )

        # Encode from nspec_dim dimensions into output (latent) dimensions
        self.encode_output_layer = ks.layers.Dense(
            self.n_latents,
            kernel_regularizer=self.kernel_regulariser_cls,
            use_bias=False,
        )

    @override
    @tf.function
    def call(self, inputs: "EncodeInputs") -> "EncodeOutputs":
        input_amp, input_phase, input_mask = inputs

        # Create initial input layer
        if self.architecture == "dense":
            x = cast("tf.Tensor", ks.layers.concatenate([input_amp, input_phase]))
        else:
            x = tf.expand_dims(input_amp, axis=-1)

        # Encode from input layers to intermediate dimensions
        for i, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x)
            x = self.dropout_layers[i](x, training=self.training)
            x = self.batch_normalisation_layers[i](x)

        # Reshape convolutional layer
        if self.architecture == "convolutional":
            x = ks.layers.Reshape((
                x.shape[-3],
                x.shape[-2] * x.shape[-1],
            ))(x)
            x = cast("tf.Tensor", ks.layers.concatenate([x, input_phase]))

        # Encode from intermediate dimensions to nspec_dim dimensions
        x = self.encode_nspec_layer(x)

        # Encode from nspec_dim dimensions to output (latent) dimensions
        x = self.encode_output_layer(x)

        is_kept = tf.cast(tf.reduce_min(input_mask, axis=-1, keepdims=True), tf.float32)
        outputs = tf.reduce_sum(x * is_kept, axis=-2) / tf.maximum(
            tf.reduce_sum(is_kept, axis=-2), y=1
        )

        outputs = tf.multiply(outputs, self.stage_mask)

        if self.training:
            is_kept = tf.reduce_max(is_kept[..., 0], axis=-1)
            reduced_is_kept = tf.reduce_sum(is_kept)
            batch_mean = tf.reduce_sum(outputs * is_kept, axis=0) / reduced_is_kept
            outputs = tf.subtract(outputs, batch_mean)
        else:
            outputs = tf.subtract(outputs, self.moving_means)

        outputs: EncodeOutputs = tf.multiply(outputs, self.stage_mask)
        return outputs


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEDecoder(ks.layers.Layer):
    def __init__(self, config: "PAEModel", **kwargs: Any) -> None:
        super().__init__(name=f"{config.name.split()[-1]}Decoder", **kwargs)

        options = cast("TFPAEModelConfig", config.options)

        # --- Config Params ---
        self.architecture: Literal["dense", "convolutional"] = options.architecture

        self.nspec_dim: int = int(config.nspec_dim)
        self.wl_dim: int = int(config.wl_dim)
        self.phase_dim: int = int(config.phase_dim)

        self.n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents
        self.n_latents: int = self.n_physical + self.n_zs

        self.decode_dims: list[int] = options.decode_dims
        self.n_decode_dims: int = len(self.decode_dims)

        self.activation_fn: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn
        self.kernel_regulariser_cls: ks.regularizers.Regularizer = (
            options.kernel_regulariser_cls(options.kernel_regulariser_penalty)
        )

        self.batch_normalisation: bool = options.batch_normalisation

        colourlaw = options.colourlaw
        if colourlaw is not None:
            _, colourlaw = np.loadtxt(colourlaw, unpack=True)
            colourlaw = tf.convert_to_tensor(colourlaw)
        self.colourlaw: tf.Tensor | None = colourlaw

        # --- Training Params ---
        self.training: bool

        # --- Tensor Specs ---
        self.input_latents_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, self.n_latents), dtype=tf.float32, name="latents"
        )
        self.input_phase_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, self.phase_dim), dtype=tf.float32, name="phase"
        )
        self.input_mask_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, self.wl_dim), dtype=tf.int32, name="mask"
        )
        self.output_spec: tf.TensorSpec = tf.TensorSpec(
            shape=(self.nspec_dim, 1), dtype=tf.float32, name="output"
        )

        # --- Layers ---
        self.decode_nspec_layer: ks.layers.Dense
        self.decode_layers: list[ks.layers.Dense | ks.layers.Conv2DTranspose]
        self.batch_normalisation_layers: list[
            ks.layers.BatchNormalization | ks.layers.Identity
        ]
        self.decode_output_layer: ks.layers.Dense | ks.layers.Reshape
        self.colourlaw_layer: ks.layers.Dense | ks.layers.Identity

    @override
    @tf.function
    def build(self, input_shape) -> None:
        # Project from input dimensions into nspec_dim dimensions
        self.decode_nspec_layer = ks.layers.Dense(
            self.nspec_dim,
            activation=self.activation_fn,
            kernel_regularizer=self.kernel_regulariser_cls,
        )

        # Decode from nspec_dim dimensions into intermediate dimensions
        self.decode_layers = []
        self.batch_normalisation_layers = []
        if self.architecture == "convolutional":
            self.decode_layers.append(
                ks.layers.Dense(
                    self.x_shape[-2] * self.x_shape[-1],
                    activation=self.activation_fn,
                    kernel_regularizer=self.kernel_regulariser_cls,
                )
            )
            self.decode_layers.append(
                ks.layers.Reshape((
                    self.phase_dim,
                    self.x_shape[-2],
                    self.x_shape[-1],
                ))
            )
        for i in range(self.n_decode_dims):
            n = self.decode_dims[i]
            if self.architecture == "dense":
                self.decode_layers.append(
                    ks.layers.Dense(
                        n,
                        activation=self.activation_fn,
                        kernel_regularizer=self.kernel_regulariser_cls,
                    )
                )
            else:
                self.decode_layers.append(
                    ks.layers.Conv2DTranspose(
                        n,
                        kernel_size=(1, self.kernel_size),
                        strides=(1, self.stride),
                        activation=self.activation_fn,
                        kernel_regularizer=self.kernel_regulariser_cls,
                        padding="same",
                    )
                )

            # Batch normalisation layer
            if self.batch_normalisation:
                self.batch_normalisation_layers.append(ks.layers.BatchNormalization())
            else:
                self.batch_normalisation_layers.append(ks.layers.Identity())

        # Decode from intermediate dimensions to output dimensions
        if self.architecture == "dense":
            self.decode_output_layer = ks.layers.Dense(
                self.wl_dim,
                kernel_regularizer=self.kernel_regulariser_cls,
            )
        else:
            self.decode_output_layer = ks.layers.Reshape((
                self.phase_dim,
                self.x_shape[-2] * self.x_shape[-1],
            ))

        # Colourlaw
        if self.n_physical > 0:
            if self.colourlaw is None:
                self.colourlaw_layer = ks.layers.Dense(
                    self.wl_dim,
                    kernel_initializer=None,
                    use_bias=False,
                    trainable=True,
                    kernel_constraint=None,
                )
            else:
                self.colourlaw_layer = ks.layers.Dense(
                    self.wl_dim,
                    kernel_initializer=tf.constant_initialiser(self.colourlaw),
                    use_bias=False,
                    trainable=False,
                    kernel_constraint=ks.constraints.NonNeg(),
                )
        else:
            self.colourlaw_layer = ks.layers.Identity()

    @override
    @tf.function
    def call(self, inputs: "DecodeInputs") -> "DecodeOutputs":
        input_latents, input_phase, input_mask = inputs

        # Repeat latent vector to match nspec_dim
        latents = tf.tile(tf.expand_dims(input_latents, axis=0), (self.nspec_dim, 1))

        # Extract physical parameters (if applicable)
        if self.n_physical > 0:
            delta_av_latent = latents[..., 0:1]
            zs_latent = latents[..., 1 : self.n_zs + 1]
            delta_m_latent = latents[..., self.n_zs + 1, self.n_zs + 2]
            delta_p_latent = latents[..., self.n_zs + 2 : self.n_zs + 3]

            # Apply Δ𝓅 shift
            input_phase += delta_p_latent
        else:
            delta_av_latent = tf.zeros((self.nspec_dim, 0))
            zs_latent = latents[..., 0:]
            delta_m_latent = tf.zeros((self.nspec_dim, 0))
            delta_p_latent = tf.zeros((self.nspec_dim, 0))

        x = ks.layers.concatenate([zs_latent, input_phase])

        # Decode from input (latent) dimensions to nspec_dim dimensions
        x = self.decode_nspec_layer(x)

        # Decode from nspec_dim dimensions to intermediate dimensions
        for i, decode_layer in enumerate(self.decode_layers):
            x = decode_layer(x)
            x = self.batch_normalisation_layers[i](x)

        # Decode from intermediate dimensions to output dimension
        outputs = self.decode_output_layer(x)

        # Apply ΔAᵥ / Δℳ  shift
        colourlaw = self.colourlaw_layer(delta_av_latent)
        if self.n_physical > 0:
            outputs *= tf.pow(10.0, -0.4 * tf.multiply(colourlaw, delta_m_latent))

        if not self.training:
            outputs = tf.nn.relu(outputs)

        # Zero out masked elements
        outputs: DecodeOutputs = tf.multiply(
            outputs,
            tf.cast(tf.reduce_max(input_mask, axis=-1, keepdims=True), tf.float32),
        )
        return outputs


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEModel(ks.Model):
    def __init__(
        self,
        config: "PAEModel",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{config.name.split()[-1]}PAEModel", **kwargs)
        self.loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss")
        self.mae_metric: ks.metrics.Metric = ks.metrics.MeanAbsoluteError(name="mae")

        options = cast("TFPAEModelConfig", config.options)
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose

        self.n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents
        self.n_latents: int = self.n_physical + self.n_zs

        self.optimiser_cls: type[ks.optimizers.Optimizer] = options.optimiser_cls
        self.loss_cls: type[ks.losses.Loss] = options.loss_fn

        self.batch_size: int = options.batch_size

        self.encoder: TFPAEEncoder = TFPAEEncoder(config)
        self.decoder: TFPAEDecoder = TFPAEDecoder(config)

        self.stage: Stage

    @override
    @tf.function
    def call(
        self,
        inputs: "EncodeInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "DecodeOutputs":
        return self.decoder((self.encoder(inputs), *inputs[1:]))

    @override
    @tf.function
    def train_step(self, data: "TensorCompatible") -> dict[str, Any]:
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x: EncodeInputs
        y: DecodeOutputs
        x, y = data

        with tf.GradientTape() as tape:
            y_pred: DecodeOutputs = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars, strict=False))
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    def setup(self, stage: "Stage") -> None:
        self.stage = stage

        # --- Compile ---
        self.compile(
            optimizer=self.optimiser_cls(learning_rate=self.stage.learning_rate),
            loss=self.loss_cls(),
        )

        # --- Setup Encoder ---
        self.encoder.training = self.stage.training
        # Mask latents which aren't being trained
        # The latents are ordered by training stage
        # ΔAᵥ -> zs -> Δℳ  -> Δ𝓅
        masked_latents = tf.zeros(self.n_latents - self.stage.stage)
        unmasked_latents = tf.ones(self.stage.stage)
        self.encoder.stage_mask = tf.concat((unmasked_latents, masked_latents), axis=0)

        # --- Setup Decoder ---
        self.decoder.training = self.stage.training

    def run(
        self, *, train_data: "SNPAEData", test_data: "SNPAEData", val_data: "SNPAEData"
    ) -> None:
        train_dataset = tf.data.Dataset.from_tensor_slices((
            (train_data.amplitude, train_data.phase, train_data.mask),
            train_data.amplitude,
        ))
        test_dataset = tf.data.Dataset.from_tensor_slices((
            (test_data.amplitude, test_data.phase, test_data.mask),
            test_data.amplitude,
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            (val_data.amplitude, val_data.phase, val_data.mask),
            val_data.amplitude,
        ))

        # --- Fit ---
        self.fit(
            train_dataset,
            epochs=self.stage.epochs,
            verbose=1 if self.verbose else 0,
            validation_data=val_dataset,
        )
