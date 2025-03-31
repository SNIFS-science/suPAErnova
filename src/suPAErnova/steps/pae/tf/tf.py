# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, Any, cast, final, override

import keras
import tensorflow as tf
from tensorflow import keras as ks

if TYPE_CHECKING:
    from suPAErnova.steps.pae.model import PAEModel
    from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig

    AmplitudeInput = tf.Tensor
    PhaseInput = tf.Tensor
    MaskInput = tf.Tensor

    DeltaAvLatent = tf.Tensor
    DeltaAmpLatent = tf.Tensor
    DeltaPeakLatent = tf.Tensor
    ZsLatent = tf.Tensor

    EncodeInputs = tuple[AmplitudeInput, PhaseInput, MaskInput]
    EncodeOutputs = tf.Tensor

    DecodeInputs = tuple[EncodeOutputs, PhaseInput, MaskInput]
    DecodeOutputs = tf.Tensor

    from tensorflow._aliases import TensorCompatible


@final
@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEEncoder(ks.layers.Layer):
    def __init__(
        self,
        config: "PAEModel",
        name: str = "encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.options = cast("TFPAEModelConfig", config.options)

        # --- Config Params ---
        self.architecture = self.options.architecture

        self.nspec_dim = config.nspec_dim
        self.wl_dim = config.wl_dim
        self.phase_dim = config.phase_dim

        self.encode_dims = self.options.encode_dims
        self.n_encode_dims = len(self.encode_dims)

        self.activation = self.options.activation_fn
        self.kernel_regulariser = self.options.kernel_regulariser_cls(
            self.options.kernel_regulariser_penalty
        )

        self.dropout = self.options.dropout
        self.batch_normalisation = self.options.batch_normalisation

        self.n_physical = 3 if self.options.physical_latents else 0
        self.n_zs = self.options.n_z_latents
        self.n_latents = self.n_physical + self.n_zs

        # --- Training Params ---
        self.training: bool
        self.stage: int
        self.stage_mask: tf.Tensor
        self.moving_means: tf.Tensor

        self.gen_tensor_specs()
        self.gen_layers()

    def gen_tensor_specs(self) -> None:
        self.input_amp_spec = tf.TensorSpec(
            shape=(self.nspec_dim, self.wl_dim), dtype=tf.float32, name="amplitude"
        )
        self.input_phase_spec = tf.TensorSpec(
            shape=(self.nspec_dim, self.phase_dim), dtype=tf.float32, name="phase"
        )
        self.input_mask_spec = tf.TensorSpec(
            shape=(self.nspec_dim, self.wl_dim), dtype=tf.int32, name="mask"
        )

    def gen_layers(self) -> None:
        # Project from input layer dimensions into intermediate dimensions
        self.projection_layers: list[ks.layers.Layer[tf.Tensor, tf.Tensor]] = []
        for i in range(self.n_encode_dims):
            n = self.encode_dims[i]
            if self.architecture == "dense":
                self.projection_layers.append(
                    ks.layers.Dense(
                        n,
                        activation=self.activation,
                        kernel_regularizer=self.kernel_regulariser,
                    )
                )
            else:
                self.projection_layers.append(
                    ks.layers.Conv2D(
                        n,
                        kernel_size=(1, self.kernel_size),
                        strides=(1, self.stride),
                        activation=self.activation,
                        kernel_regularizer=self.kernel_regulariser,
                        padding="same",
                    )
                )

        # Dropout layer
        if self.dropout > 0:
            self.dropout_layer = ks.layers.Dropout(
                self.dropout, noise_shape=[None, 1, None]
            )
        else:
            self.dropout_layer = ks.layers.Identity()

        # Batch normalisation layer
        if self.batch_normalisation:
            self.batch_normalisation_layer = ks.layers.BatchNormalization()
        else:
            self.batch_normalisation_layer = ks.layers.Identity()

        # Project from intermediate dimensions into nspec_dim dimensions
        self.project_nspec_layer = ks.layers.Dense(
            self.nspec_dim,
            activation=self.activation,
            kernel_regularizer=self.kernel_regulariser,
        )

        # Project from nspec_dim dimensions into output dimensions
        self.project_output_layer = ks.layers.Dense(
            self.n_latents,
            kernel_regularizer=self.kernel_regulariser,
            use_bias=False,
        )

    def setup(self, *, training: bool, stage: int) -> None:
        self.training = training
        self.stage = stage

        # Mask latents which aren't being trained
        # The latents are ordered by training stage
        # ΔAᵥ -> zs -> Δℳ  -> Δ𝓅
        masked_latents = tf.zeros(self.n_latents - self.stage)
        unmasked_latents = tf.ones(self.stage)
        self.stage_mask = tf.concat((unmasked_latents, masked_latents), axis=0)

    @override
    def call(self, inputs: "EncodeInputs") -> "EncodeOutputs":
        input_amp, input_phase, input_mask = inputs
        tf.ensure_shape(input_amp, self.input_amp_spec)
        tf.ensure_shape(input_phase, self.input_phase_spec)
        tf.ensure_shape(input_mask, self.input_mask_spec)

        # Create initial input layer
        if self.architecture == "dense":
            x = cast("tf.Tensor", ks.layers.concatenate([input_amp, input_phase]))
        else:
            x = tf.expand_dims(input_amp, axis=-1)

        # Project to intermediate dimensions
        for projection_layer in self.projection_layers:
            x = projection_layer(x)
            x = self.dropout_layer(x)
            x = self.batch_normalisation_layer(x)

        # Reshape convolutional layer
        if self.architecture == "convolutional":
            x_shape = x.shape
            x = ks.layers.Reshape((
                x_shape[-3],
                cast("int", x_shape[-2]) * cast("int", x_shape[-1]),
            ))(x)
            x = cast("tf.Tensor", ks.layers.concatenate([x, input_phase]))

        # Project to nspec_dim
        x = self.project_nspec_layer(x)

        # Project to output dimensions (self.n_latents)
        x = self.project_output_layer(x)

        is_kept = tf.reduce_min(input_mask, axis=-1)
        outputs: EncodeOutputs = tf.reduce_sum(x * is_kept, axis=-2) / tf.maximum(
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

        return tf.multiply(outputs, self.stage_mask)


@final
@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEDecoder(ks.layers.Layer):
    def __init__(
        self, config: "PAEModel", name: str = "decoder", **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

    @override
    def call(self, inputs: "DecodeInputs") -> "DecodeOutputs":
        pass


@final
@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEModel(ks.Model):
    def __init__(
        self,
        config: "PAEModel",
        name: str = "autoencoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.encoder = TFPAEEncoder(config)
        self.decoder = TFPAEDecoder(config)

    def setup(self, *, training: bool, stage: int) -> None:
        self.encoder.setup(training=training, stage=stage)

    @override
    def call(
        self,
        inputs: "EncodeInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "DecodeOutputs":
        amp, phase, mask = inputs
        latents = self.encoder((amp, phase, mask))
        return self.decoder((latents, phase, mask))
