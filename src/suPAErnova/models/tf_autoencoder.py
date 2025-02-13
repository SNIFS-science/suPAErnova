from typing import TYPE_CHECKING, final

import keras as ks
from keras import layers
import numpy as np
import tensorflow as tf

from suPAErnova.models import PAEModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from keras import KerasTensor

    from suPAErnova.steps.model import ModelStep
    from suPAErnova.utils.typing import CFG


# Define Custom Layers
def layer(tf_fn: "Callable[[ks.KerasTensor], tf.Tensor]"):
    class WrappedLayer(layers.Layer):
        def __init__(self) -> None:
            super().__init__()

        def call(self, x: "KerasTensor", **kwargs):
            if isinstance(x, tf.SparseTensor):
                # Convert sparse tensor to dense tensor
                x = tf.sparse.to_dense(x)
            return tf_fn(x, **kwargs)

    return WrappedLayer()


reduce_max = layer(tf.reduce_max)
reduce_min = layer(tf.reduce_min)
reduce_sum = layer(tf.reduce_sum)


@final
class TFAutoencoder(ks.Model, PAEModel):
    def __init__(self, step: "ModelStep", training_params: "CFG") -> None:
        PAEModel.__init__(self, step, training_params)
        ks.Model.__init__(self)

        # Training Settings
        self.training = self.training_params["training"]
        self.train_stage = self.training_params["train_stage"]

        self.colourlaw = self.params["COLOURLAW"]

        # Network Settings
        self.layer_type = self.params["LAYER_TYPE"]
        self.activation = self.params["ACTIVATION"]
        self.lr = self.params["LR"]
        self.lr_deltat = self.params["LR_DELTAT"]
        self.lr_decay_rate = self.params["LR_DECAY_RATE"]
        self.lr_decay_steps = self.params["LR_DECAY_STEPS"]

        if self.params["KERNEL_REGULARISER"] > 0:
            self.kernel_regulariser = ks.regularizers.l2(
                self.params["KERNEL_REGULARISER"],
            )
        else:
            self.kernel_regulariser = None
        self.dropout = self.params["DROPOUT"]
        self.batchnorm = self.params["BATCHNORM"]
        self.physical_latent = self.params["PHYSICAL_LATENT"]

        # Model Dimensions
        self.time_dim = int(self.data.n_timemax)
        self.data_dim = self.data.n_wavelength
        self.cond_dim = self.params["COND_DIM"]
        self.encode_dims = [*self.params["ENCODE_DIMS"], self.time_dim]
        self.decode_dims = [self.time_dim, *self.params["DECODE_DIMS"]]
        self.latent_dim = self.params["LATENT_DIM"]
        self.num_physical_latent_dims = 3  # [delta t, delta m, Av]

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs_data = layers.Input(shape=(self.time_dim, self.data_dim))
        inputs_cond = layers.Input(shape=(self.time_dim, self.cond_dim))
        inputs_mask = layers.Input(shape=(self.time_dim, self.data_dim))

        # Add either a fully connected or convolutional block
        if self.layer_type == "DENSE":
            x = layers.concatenate([inputs_data, inputs_cond])
        else:
            x = inputs_data

        for i, n in enumerate(self.encode_dims[:-1]):
            if self.layer_type == "DENSE":
                x = layers.Dense(
                    n,
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regulariser,
                )(x)
            else:
                if i == 0:
                    x = tf.expand_dims(x, axis=-1)
                x = layers.Conv2D(
                    n,
                    kernel_size=(1, self.kernel_size),
                    strides=(1, self.stride),
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regulariser,
                    padding="same",
                    input_shape=(self.time_dim, self.data_dim) if i == 0 else None,
                )(x)

            if self.dropout > 0:
                x = layers.Dropout(self.dropout, noise_shape=[None, 1, None])(
                    x,
                    training=self.training,
                )

            if self.batchnorm:
                x = layers.BatchNormalization()(x)

        if self.layer_type == "CONVOLUTION":
            # Reshape to pass to Dense layers
            self.x_shape = x.shape
            x = layers.Reshape((
                self.x_shape[-3],
                self.x_shape[-2] * self.x_shape[-1],
            ))(x)
            x = layers.concatenate([x, inputs_cond])

        # Dense Layers
        x = layers.Dense(
            self.encode_dims[-1],
            activation=self.activation,
            kernel_regularizer=self.kernel_regulariser,
        )(x)

        x = layers.Dense(
            self.latent_dim + self.num_physical_latent_dims,
            kernel_regularizer=self.kernel_regulariser,
            use_bias=False,
        )(x)

        # Need to mask time samples that do not exist = take mean of non-masked latent variables
        # return is_kept (N_sn, N_spectra) = 0 if any wavelength bin was masked, as bad value will effect encoding
        is_kept = reduce_min(inputs_mask, axis=-1, keepdims=True)
        outputs = reduce_sum(x * is_kept, axis=-2) / reduce_sum(is_kept, axis=-2)

        if self.physical_latent:
            dtime = outputs[..., 0:1]
            amplitude = outputs[..., 1:2]
            colour = outputs[..., 2:3]
            latent = outputs[..., 3:]

            if self.train_stage == 0:
                latent *= 0.0
                amplitude *= 0.0
                dtime *= 0.0

            if 0 < self.train_stage <= self.latent_dim:
                # Construct Mask
                latent_train_mask = np.ones(self.latent_dim, dtype=np.float32)
                latent_train_mask[self.train_stage :] = 0.0
                latent_train_mask = tf.convert_to_tensor(latent_train_mask)

                # Apply Mask
                latent *= latent_train_mask
                amplitude *= 0.0
                dtime *= 0.0

            if self.train_stage == self.latent_dim + 1:
                dtime *= 0.0

            # Make dtime, amplitude, and colour of non-masked SN have mean 0
            if self.training:
                is_kept = reduce_max(is_kept[..., 0], axis=-1)
                reduced_is_kept = reduce_sum(is_kept)

                print(dtime, is_kept, dtime * is_kept)
                batch_mean_dtime = reduce_sum(dtime * is_kept, axis=0) / reduced_is_kept
                dtime = layers.subtract([dtime, batch_mean_dtime])

                batch_mean_amplitude = (
                    reduce_sum(amplitude * is_kept, axis=0) / reduced_is_kept
                )
                amplitude = layers.subtract([amplitude, batch_mean_amplitude])

                batch_mean_colour = (
                    reduce_sum(colour * is_kept, axis=0) / reduced_is_kept
                )
                colour = layers.subtract([colour, batch_mean_colour])

            else:
                dtime = layers.subtract([
                    dtime,
                    tf.Variable([self.bn_moving_means[0]]),
                ])

                amplitude = layers.subtract([
                    amplitude,
                    tf.Variable([self.bn_moving_means[1]]),
                ])

                colour = layers.subtract([
                    colour,
                    tf.Variable([self.bn_moving_means[3]]),
                ])
            outputs = layers.concatenate([dtime, amplitude, colour, latent])

        inputs = [inputs_data, inputs_cond, inputs_mask]

        return ks.Model(inputs=inputs, outputs=outputs)

    def build_decoder(self):
        inputs_latent = layers.Input(
            shape=(self.latent_dim + self.num_physical_latent_dims,),
            name="latent_params",
        )
        inputs_cond = layers.Input(
            shape=(self.time_dim, self.cond_dim),
            name="conditional_params",
        )
        inputs_mask = layers.Input(shape=(self.time_dim, self.data_dim))

        # Repeate latent vector to match number of data timesteps
        latent = layers.RepeatVector(self.time_dim)(inputs_latent)

        # Set up physical latent space (if desired)
        if self.physical_latent:
            dtime = latent[..., 0:1]
            amplitude = latent[..., 1:2]
            colour = latent[..., 2:3]

            latent = latent[..., self.num_physical_latent_dims :]

            # Concatenate physical (non time-varying) parameters
            x = layers.concatenate([latent, inputs_cond + dtime])
        else:
            x = layers.concatenate([latent, inputs_cond])

        x = layers.Dense(
            self.decode_dims[0],
            activation=self.activation,
            kernel_regularizer=self.kernel_regulariser,
        )(x)

        for i, n in enumerate(self.decode_dims[1:]):
            if self.layer_type == "DENSE":
                # Fully connected network
                x = layers.Dense(
                    n,
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regulariser,
                )(x)
            else:
                if i == 0:
                    x = layers.Dense(
                        self.x_shape[-2] * self.x_shape[-1],
                        activation=self.activation,
                        kernel_regularizer=self.kernel_regulariser,
                    )(x)
                    x = layers.Reshape((
                        self.time_dim,
                        self.x_shape[-2],
                        self.x_shape[-1],
                    ))(x)
                x = layers.Conv2DTranspose(
                    n,
                    kernel_size=(1, self.kernel_size),
                    strides=(1, self.stride),
                    activation=self.activation,
                    padding="same",
                    kernel_regularizer=self.kernel_regulariser,
                )(x)
            if self.batchnorm:
                x = layers.BatchNormalization()(x)

        if self.layer_type == "DENSE":
            outputs = layers.Dense(
                self.data_dim,
                kernel_regularizer=self.kernel_regulariser,
            )(x)
        else:
            outputs = layers.Reshape((self.time_dim, x.shape[-2] * x.shape[-1]))(x)

        if self.physical_latent:
            colourlaw = layers.Dense(
                self.data_dim,
                kernel_initializer=None
                if self.colourlaw is None
                else tf.constant_initializer(self.colourlaw),
                name="colour_law",
                use_bias=False,
                trainable=self.colourlaw is None,
                kernel_constraint=None
                if self.colourlaw is None
                else ks.constraints.NonNeg(),
            )(colour)

            outputs *= 10 ** (-0.4 * (colourlaw * amplitude))
        if not self.training:
            outputs = tf.nn.relu(outputs)

        inputs = [inputs_latent, inputs_cond, inputs_mask]
        # Zero spectra that do not exist
        outputs *= reduce_max(inputs_mask, axis=-1, keepdims=True)

        return ks.Model(inputs=inputs, outputs=outputs)

    def encode(self, x, cond, mask):
        return self.encoder((x, cond, mask))

    def decode(self, x, cond, mask):
        return self.decoder((x, cond, mask))

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
