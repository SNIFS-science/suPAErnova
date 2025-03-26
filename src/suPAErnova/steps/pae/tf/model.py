# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, Any, Literal, cast, final, override

import keras
from pydantic import (
    BaseModel,
    PositiveInt,  # noqa: TC002
)
import tensorflow as tf
from tensorflow import keras as ks

if TYPE_CHECKING:
    AmplitudeInput = tf.Tensor
    PhaseInput = tf.Tensor
    MaskInput = tf.Tensor

    DeltaAvLatent = tf.Tensor
    DeltaAmpLatent = tf.Tensor
    DeltaPeakLatent = tf.Tensor
    ZLatent = tf.Tensor
    Latent = DeltaAvLatent | DeltaAmpLatent | DeltaPeakLatent | ZLatent

    EncodeInputs = tuple[AmplitudeInput, PhaseInput, MaskInput]
    EncodeOutputs = tuple[Latent, ...]

    DecodeInputs = tuple[EncodeOutputs, PhaseInput, MaskInput]
    DecodeOutputs = tf.Tensor


class TFPAEModelConfig(BaseModel):
    # Required
    architecture: Literal["dense", "convolutional"]
    encode_dims: list["PositiveInt"]

    nspec_dim: "PositiveInt"
    wl_dim: "PositiveInt"

    # Optional
    phase_dim: "PositiveInt" = 1


@final
@keras.saving.register_keras_serializable()
class TFPAEEncoder(ks.layers.Layer):
    def __init__(
        self,
        config: TFPAEModelConfig,
        name: str = "encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.architecture = config.architecture

        self.nspec_dim = config.nspec_dim
        self.wl_dim = config.wl_dim
        self.phase_dim = config.phase_dim

        self.encode_dims = config.encode_dims
        self.n_encode_dims = len(self.encode_dims)

        # --- TensorSpecs ---
        self.input_amp_spec = tf.TensorSpec(
            shape=(self.nspec_dim, self.wl_dim), dtype=tf.float32, name="amplitude"
        )
        self.input_phase_spec = tf.TensorSpec(
            shape=(self.nspec_dim, self.phase_dim), dtype=tf.float32, name="phase"
        )
        self.input_mask_spec = tf.TensorSpec(
            shape=(self.nspec_dim, self.wl_dim), dtype=tf.int32, name="mask"
        )
        self.input_spec: tuple[tf.TensorSpec, tf.TensorSpec, tf.TensorSpec] = (
            self.input_amp_spec,
            self.input_phase_spec,
            self.input_mask_spec,
        )

        # --- Layers ---
        # Project from input layer dimensions into intermediate dimensions
        self.projection_layers: list[ks.layers.Layer] = []
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
            self.n_zs + self.n_physical,
            kernel_regularizer=self.kernel_regulariser,
            use_bias=False,
        )

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
            x = ks.layers.Reshape((x_shape[-3], x_shape[-2] * x_shape[-1]))(x)
            x = cast("tf.Tensor", ks.layers.concatenate([x, input_phase]))

        # Project to nspec_dim
        x = self.project_nspec_layer(x)

        # Project to output dimensions (n_zs + n_physical)
        x = self.project_output_layer(x)

        return ()


@final
@keras.saving.register_keras_serializable()
class TFPAEDecoder(ks.layers.Layer):
    def __init__(
        self, config: TFPAEModelConfig, name: str = "decoder", **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

    @override
    def call(self, inputs: "DecodeInputs") -> "DecodeOutputs":
        pass


@final
@keras.saving.register_keras_serializable()
class TFPAEModel(ks.layers.Layer):
    def __init__(
        self,
        config_dict: dict[str, Any],
        name: str = "autoencoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        config = TFPAEModelConfig.model_validate(config_dict)

        self.encoder = TFPAEEncoder(config)
        self.decoder = TFPAEDecoder(config)

    @override
    def call(self, inputs: "EncodeInputs") -> "DecodeOutputs":
        latents = self.encoder(inputs)
        return self.decoder(latents)
