from typing import TYPE_CHECKING, final

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import layers

from suPAErnova.models import PAEModel
from suPAErnova.model_utils.tf_layers import (
    relu,
    maximum,
    reduce_max,
    reduce_min,
    reduce_sum,
)

if TYPE_CHECKING:
    from suPAErnova.steps.model import ModelStep
    from suPAErnova.utils.suPAErnova_types import CFG


@final
class TFAutoencoder(ks.Model, PAEModel):
    def __init__(self, step: "ModelStep", training_params: "CFG") -> None:
        PAEModel.__init__(self, step, training_params)
        ks.Model.__init__(self)

        # Training Settings
        print(self.training_params)
        self.training = self.training_params["training"]
        self.train_stage = self.training_params["train_stage"]
        self.learning_rate = self.training_params["learning_rate"]
        self.moving_means = self.training_params.get("moving_means", [0.0, 0.0, 0.0])

        self.colourlaw = self.params["COLOURLAW"]
        self.loss_fn = self.params["LOSS"].upper()
        self.loss_amplitude_offset = self.params["LOSS_AMPLITUDE_OFFSET"]
        self.loss_amplitude_parameter = self.params["LOSS_AMPLITUDE_PARAMETER"]
        self.loss_covariance = self.params["LOSS_COVARIANCE"]
        self.decorrelate_dust = self.params["DECORRELATE_DUST"]
        self.decorrelate_all = self.params["DECORRELATE_ALL"]
        self.save_model = self.params["SAVE_MODEL"]
        self.load_best = self.params["LOAD_BEST"]

        # Network Settings
        self.layer_type = self.params["LAYER_TYPE"].upper()
        self.activation = self.params["ACTIVATION"]

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
        self.n_spectra = int(self.data.n_timemax)
        self.n_flux = self.data.n_wavelength
        self.cond_dim = self.params["COND_DIM"]
        self.encode_dims = [*self.params["ENCODE_DIMS"], self.n_spectra]
        self.decode_dims = [self.n_spectra, *self.params["DECODE_DIMS"]]
        self.latent_dim = self.params["LATENT_DIM"]
        self.num_physical_latent_dims = 3  # [delta t, delta m, Av]

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        # Input layers

        # Input spectra from a single supernovae
        inputs_flux = layers.Input(
            shape=(self.n_spectra, self.n_flux),
            dtype=tf.float32,
        )
        # Phase-based conditional layer
        inputs_time = layers.Input(
            shape=(self.n_spectra, self.cond_dim),
            dtype=tf.float32,
        )
        # Mask input layer
        inputs_mask = layers.Input(shape=(self.n_spectra, self.n_flux), dtype=tf.int32)

        if self.layer_type == "DENSE":
            x = layers.concatenate([inputs_flux, inputs_time])
        else:
            x = inputs_flux

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
                    input_shape=(self.n_spectra, self.n_flux) if i == 0 else None,
                )(x)

            if self.dropout > 0:
                x = layers.Dropout(self.dropout, noise_shape=[None, 1, None])(
                    x,
                    training=self.training,
                )

            if self.batchnorm:
                x = layers.BatchNormalization()(x)

        if self.layer_type == "CONVOLUTION":
            self.x_shape = x.shape
            x = layers.Reshape((
                self.x_shape[-3],
                self.x_shape[-2] * self.x_shape[-1],
            ))(x)
            x = layers.concatenate([x, inputs_time])

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

        is_kept = reduce_min(inputs_mask, axis=-1, keepdims=True)
        outputs = reduce_sum(x * is_kept, axis=-2) / maximum(
            reduce_sum(is_kept, axis=-2),
            y=1,
        )

        if self.physical_latent:
            dtime = outputs[..., 0:1]
            amplitude = outputs[..., 1:2]
            colour = outputs[..., 2:3]
            latent = outputs[..., 3:]

            #
            # Set the parameters you're not training to 0
            #

            # Initial training, only train colour
            if self.train_stage == 0:
                latent *= 0.0
                amplitude *= 0.0
                dtime *= 0.0

            # Latent training, train the first `train_stage` latent parameters, and colour
            if 0 < self.train_stage <= self.latent_dim:
                # Construct Mask
                latent_train_mask = np.ones(self.latent_dim)
                latent_train_mask[self.train_stage :] = 0.0
                latent_train_mask = tf.convert_to_tensor(latent_train_mask)

                # Apply Mask
                latent *= latent_train_mask
                amplitude *= 0.0
                dtime *= 0.0

            # Train colour, all latent dims, and amplitude
            if self.train_stage == self.latent_dim + 1:
                dtime *= 0.0

            # if self.train_stage > self.latent_dim + 1 then train all parameters

            # Make dtime, amplitude, and colour of non-masked SN have mean 0
            if self.training:
                is_kept = reduce_max(is_kept[..., 0], axis=-1)
                reduced_is_kept = reduce_sum(is_kept)

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
                dtime -= self.moving_means[0]
                amplitude -= self.moving_means[1]
                colour -= self.moving_means[2]
            outputs = layers.concatenate([dtime, amplitude, colour, latent])

        inputs = [inputs_flux, inputs_time, inputs_mask]

        return ks.Model(inputs=inputs, outputs=outputs)

    def build_decoder(self):
        inputs_latent = layers.Input(
            shape=(self.latent_dim + self.num_physical_latent_dims,),
            name="latent_params",
        )
        inputs_time = layers.Input(
            shape=(self.n_spectra, self.cond_dim),
            name="conditional_params",
        )
        inputs_mask = layers.Input(shape=(self.n_spectra, self.n_flux))

        # Repeate latent vector to match number of data timesteps
        latent = layers.RepeatVector(self.n_spectra)(inputs_latent)

        # Set up physical latent space (if desired)
        if self.physical_latent:
            dtime = latent[..., 0:1]
            amplitude = latent[..., 1:2]
            colour = latent[..., 2:3]

            latent = latent[..., self.num_physical_latent_dims :]

            # Concatenate physical (non time-varying) parameters
            x = layers.concatenate([latent, inputs_time + dtime])
        else:
            x = layers.concatenate([latent, inputs_time])

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
                        self.n_spectra,
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
                self.n_flux,
                kernel_regularizer=self.kernel_regulariser,
            )(x)
        else:
            outputs = layers.Reshape((self.n_spectra, x.shape[-2] * x.shape[-1]))(x)

        if self.physical_latent:
            colourlaw = layers.Dense(
                self.n_flux,
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
            outputs = relu(outputs)

        inputs = [inputs_latent, inputs_time, inputs_mask]
        # Zero spectra that do not exist
        outputs *= reduce_max(inputs_mask, axis=-1, keepdims=True)

        return ks.Model(inputs=inputs, outputs=outputs)

    def encode(self, flux, time, mask):
        return self.encoder((flux, time, mask))

    def decode(self, z, time, mask):
        return self.decoder((z, time, mask))
