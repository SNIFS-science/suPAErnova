# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    cast,
    override,
)

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

if TYPE_CHECKING:
    from typing import (
        Any,
        Literal,
        Annotated,
    )
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable, Sequence

    from numpy import typing as npt

    from suPAErnova.steps.pae.model import Stage, PAEModel
    from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig

    # === Custom Types ===
    S = Literal
    type FTensor[Shape: str] = Annotated[tf.Tensor, tf.float32, Shape]
    type ITensor[Shape: str] = Annotated[tf.Tensor, tf.int32, Shape]
    type FRTensor[Shape: str] = Annotated[tf.RaggedTensor, tf.float32, Shape]
    type IRTensor[Shape: str] = Annotated[tf.RaggedTensor, tf.int32, Shape]
    type Tensor[Shape: str] = FTensor[Shape] | ITensor[Shape]
    type RaggedTensor[Shape: str] = FRTensor[Shape] | IRTensor[Shape]
    type GenericTensor[Shape: str] = Tensor[Shape] | RaggedTensor[Shape]

    type TensorCompatible = (
        tf.Tensor
        | str
        | float
        | np.ndarray[Any, Any]
        | np.number[Any]
        | Sequence[TensorCompatible]
    )

    # --- Encoder Tensors ---
    EncoderInputsShape = tuple[tuple[int, int, int], tuple[int, int, int]]
    EncoderInputs = tuple[
        FTensor[S["batch_dim nspec_dim phase_dim"]],
        FTensor[S["batch_dim nspec_dim wl_dim"]],
    ]
    EncoderOutputs = FTensor[S["batch_dim nspec_dim n_latents"]]

    # --- Decoder Tensors ---
    DecoderInputsShape = tuple[tuple[int, int, int], tuple[int, int, int]]
    type DecoderInputs = tuple[
        FTensor[S["batch_dim nspec_dim n_latents"]],
        FTensor[S["batch_dim nspec_dim phase_dim"]],
    ]
    DecoderOutputs = FTensor[S["batch_dim nspec_dim wl_dim"]]

    # --- Model Tensors ---
    RawData = tuple[
        npt.NDArray[np.float32],  # phase
        npt.NDArray[np.float32],  # dphase
        npt.NDArray[np.float32],  # amp
        npt.NDArray[np.float32],  # damp
        npt.NDArray[np.int32],  # mask
    ]
    PrepData = tuple[
        ITensor[S["n_sn nspec_dim"]],
        IRTensor[S["n_sn n_spec_to_shuffle(sn)"]],
        IRTensor[S["n_sn n_spec_to_shuffle(sn)"]],
    ]

    EpochInputs = tuple[
        tuple[
            FTensor[S["batch_dim nspec_dim phase_dim"]],  # phase
            FTensor[S["batch_dim nspec_dim phase_dim"]],  # dphase
            FTensor[S["batch_dim nspec_dim wl_dim"]],  # amp
            FTensor[S["batch_dim nspec_dim wl_dim"]],  # damp
            ITensor[S["batch_dim nspec_dim wl_dim"]],  # mask
        ],
        tuple[
            ITensor[S["batch_dim nspec_dim"]],
            IRTensor[S["batch_dim n_spec_to_shuffle(sn)"]],
            IRTensor[S["batch_dim n_spec_to_shuffle(sn)"]],
        ],
    ]

    ModelInputs = tuple[
        FTensor[S["batch_dim nspec_dim phase_dim"]],  # phase
        FTensor[S["batch_dim nspec_dim phase_dim"]],  # dphase
        FTensor[S["batch_dim nspec_dim wl_dim"]],  # amp
        FTensor[S["batch_dim nspec_dim wl_dim"]],  # damp
        ITensor[S["batch_dim nspec_dim wl_dim"]],  # mask
    ]


class TypedLayer[
    L: "ks.layers.Layer[tf.Tensor, tf.Tensor]",
    I: "GenericTensor[str]",
    O: "GenericTensor[str]",
](ks.layers.Layer):
    def __init__(self, layer: "ks.layers.Layer[I, O]", **kwargs: "Any") -> None:
        super().__init__(**kwargs)
        self.layer: "ks.layers.Layer[I, O]" = layer

    @override
    def build(self, input_shape: "Any", *args: "Any", **kwargs: "Any") -> None:
        self.layer.build(input_shape, *args, **kwargs)

    @override
    def call(self, inputs: "I", *args: "Any", **kwargs: "Any") -> "O":
        return self.layer(inputs, *args, **kwargs)

    @override
    def __call__(self, inputs: "I", *args: "Any", **kwargs: "Any") -> "O":
        return self.layer(inputs, *args, **kwargs)


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEEncoder(ks.layers.Layer):
    def __init__(
        self,
        options: "TFPAEModelConfig",
        name: str,
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=f"{name.split()[-1]}Encoder", **kwargs)

        # --- Config Params ---
        n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        n_zs: int = options.n_z_latents
        self.n_latents: int = n_physical + n_zs
        self.latents_z_mask: ITensor[Literal["n_latents"]]
        self.latents_physical_mask: ITensor[Literal["n_latents"]]

        self.stage_num: int
        self.moving_means: FTensor[Literal["n_latents"]]

        # Mask tensors to select specific latents
        if n_physical > 0:
            self.latents_z_mask = tf.concat(
                (tf.zeros(1), tf.ones(n_zs), tf.zeros(1), tf.zeros(1)), axis=0
            )
        else:
            self.latents_z_mask = tf.ones(n_zs)
        self.latents_physical_mask = tf.abs(self.latents_z_mask - 1)  # Swap 0s and 1s

        self.encode_dims: list[int] = options.encode_dims

        self.activation: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn
        self.regulariser: ks.regularizers.Regularizer = options.kernel_regulariser_cls(
            options.kernel_regulariser_penalty
        )

        self.dropout: float = options.dropout
        self.batch_normalisation: bool = options.batch_normalisation

        # --- Layers ---
        self.encode_layers: list[
            TypedLayer[
                ks.layers.Dense,
                FTensor[Literal["batch_dim nspec_dim _"]],
                FTensor[Literal["batch_dim nspec_dim encode_dim"]],
            ]
        ] = []
        self.dropout_layers: list[
            TypedLayer[
                ks.layers.Dropout | ks.layers.Identity,
                FTensor[Literal["batch_dim nspec_dim encode_dim"]],
                FTensor[Literal["batch_dim nspec_dim encode_dim"]],
            ]
        ] = []
        self.batch_normalisation_layers: list[
            TypedLayer[
                ks.layers.BatchNormalization | ks.layers.Identity,
                FTensor[Literal["batch_dim nspec_dim encode_dim"]],
                FTensor[Literal["batch_dim nspec_dim encode_dim"]],
            ]
        ] = []
        self.encode_nspec_layer: TypedLayer[
            ks.layers.Dense,
            FTensor[Literal["batch_dim nspec_dim encode_dim"]],
            FTensor[Literal["batch_dim nspec_dim nspec_dim"]],
        ]
        self.encode_output_layer: TypedLayer[
            ks.layers.Dense,
            FTensor[Literal["batch_dim nspec_dim nspec_dim"]],
            FTensor[Literal["batch_dim nspec_dim n_latents"]],
        ]
        self.repeat_latent_layer: TypedLayer[
            ks.layers.RepeatVector,
            FTensor[Literal["batch_dim n_latents"]],
            FTensor[Literal["batch_dim nspec_dim n_latents"]],
        ]

    @override
    def build(self, input_shape: "EncoderInputsShape") -> None:
        (_batch_dim, nspec_dim, _phase_dim), (_batch_dim, _nspec_dim, _wl_dim) = (
            input_shape
        )

        # Encode from input layer dimensions into intermediate dimensions
        for encode_dim in self.encode_dims:
            self.encode_layers.append(
                TypedLayer(
                    ks.layers.Dense(
                        encode_dim,
                        activation=self.activation,
                        kernel_regularizer=self.regulariser,
                    )
                )
            )
            # Dropout layer
            self.dropout_layers.append(
                TypedLayer(
                    ks.layers.Dropout(self.dropout, noise_shape=[None, 1, None])
                    if self.dropout > 0
                    else ks.layers.Identity()
                )
            )
            # Batch normalisation layer
            self.batch_normalisation_layers.append(
                TypedLayer(
                    ks.layers.BatchNormalization()
                    if self.batch_normalisation
                    else ks.layers.Identity()
                )
            )

        # Encode from intermediate dimensions into nspec_dim dimensions
        self.encode_nspec_layer = TypedLayer(
            ks.layers.Dense(
                nspec_dim,
                activation=self.activation,
                kernel_regularizer=self.regulariser,
            )
        )

        # Encode from nspec_dim dimensions into output (latent) dimensions
        self.encode_output_layer = TypedLayer(
            ks.layers.Dense(
                self.n_latents,
                kernel_regularizer=self.regulariser,
                use_bias=False,
            )
        )

        # Repeat latent vector to match nspec_dim
        self.repeat_latent_layer = TypedLayer(ks.layers.RepeatVector(nspec_dim))

    @override
    @tf.function
    def call(
        self,
        inputs: "EncoderInputs",
        mask: "ITensor[S['batch_dim nspec_dim wl_dim']] | None" = None,
        training: bool | None = None,
    ) -> "EncoderOutputs":
        training = False if training is None else training

        input_phase = inputs[0]
        input_amp = inputs[1]
        input_mask = (
            mask if mask is not None else tf.ones_like(input_amp, dtype=tf.int32)
        )

        # Create initial input layer
        x: FTensor[S["batch_dim nspec_dim wl_dim+1"]] = ks.layers.concatenate([
            input_amp,
            input_phase,
        ])

        # Encode from input layers to intermediate dimensions
        for i, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x)
            x = self.dropout_layers[i](x, training=training)
            x = self.batch_normalisation_layers[i](x)

        # Encode from intermediate dimensions to nspec_dim dimensions
        x = self.encode_nspec_layer(x)

        # Encode from nspec_dim dimensions to output (latent) dimensions
        x = self.encode_output_layer(x)

        # Determine which spectra to keep
        is_kept = tf.cast(tf.reduce_min(input_mask, axis=-1, keepdims=True), tf.float32)
        if TYPE_CHECKING:
            is_kept = cast("FTensor[S['batch_dim nspec_dim 1']]", is_kept)

        # Latent tensor is the average of the latent values over all unmasked spectra
        # First sum latents over all unmasked spectra
        batch_sum = tf.reduce_sum(x * is_kept, axis=-2)
        if TYPE_CHECKING:
            batch_sum = cast("FTensor[S['batch_dim n_latents']]", batch_sum)

        # Then determine the number of unmasked spectra
        batch_num = tf.maximum(tf.reduce_sum(is_kept, axis=-2), y=1)
        if TYPE_CHECKING:
            batch_num = cast("FTensor[S['batch_dim 1']]", batch_num)

        # Finally, calculate the average
        latents = batch_sum / batch_num
        if TYPE_CHECKING:
            latents = cast("FTensor[S['batch_dim n_latents']]", latents)

        if training:
            # Mask latents which aren't being trained
            # The latents are ordered by training stage
            # ΔAᵥ -> zs -> Δℳ  -> Δ𝓅
            masked_latents = tf.zeros(self.n_latents - self.stage_num)
            if TYPE_CHECKING:
                masked_latents = cast(
                    "FTensor[S['n_latents-stage_num']]", masked_latents
                )
            unmasked_latents = tf.ones(self.stage_num)
            if TYPE_CHECKING:
                unmasked_latents = cast("FTensor[S['stage_num']]", unmasked_latents)
            latent_mask = tf.concat((unmasked_latents, masked_latents), axis=0)
            if TYPE_CHECKING:
                latent_mask = cast("FTensor[S['n_latents']]", latent_mask)
            latents *= latent_mask

            # Normalise the physical latents within this batch such that they have a mean of 0
            latents_sum = tf.reduce_sum(latents, axis=0, keepdims=True)
            latents_num = tf.reduce_sum(tf.ones_like(latents), axis=0, keepdims=True)
            latents_mean = self.latents_physical_mask * latents_sum / latents_num
            latents -= latents_mean
            if TYPE_CHECKING:
                latents_sum = cast("FTensor[S['1 n_latents']]", latents_sum)
                latents_num = cast("FTensor[S['1 n_latents']]", latents_num)
                latents_mean = cast("FTensor[S['1 n_latents']]", latents_mean)
        else:
            # Normalise the physical latents within this batch such that the entire unbatched sample has a mean of 0
            latents -= self.latents_physical_mask * self.moving_means
        if TYPE_CHECKING:
            latents = cast("FTensor[S['batch_dim n_latents']]", latents)

        # Repeat latent layers across NSpecDim
        encoded = self.repeat_latent_layer(latents)
        if TYPE_CHECKING:
            encoded = cast("FTensor[S['batch_dim nspec_dim n_latents']]", encoded)

        return encoded


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEDecoder(ks.layers.Layer):
    def __init__(self, options: "TFPAEModelConfig", name: str, **kwargs: "Any") -> None:
        super().__init__(name=f"{name.split()[-1]}Decoder", **kwargs)
        # --- Config Params ---
        self.n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents

        self.wl_dim: int
        self.decode_dims: list[int] = options.decode_dims

        self.activation: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn
        self.regulariser: ks.regularizers.Regularizer = options.kernel_regulariser_cls(
            options.kernel_regulariser_penalty
        )
        self.batch_normalisation: bool = options.batch_normalisation

        colourlaw = options.colourlaw
        if colourlaw is not None:
            _, colourlaw = np.loadtxt(colourlaw, unpack=True)
        self.colourlaw: npt.NDArray[np.float64] | None = colourlaw

        # --- Layers ---
        self.decode_nspec_layer: TypedLayer[
            ks.layers.Dense,
            FTensor[S["batch_dim nspec_dim _"]],
            FTensor[S["batch_dim nspec_dim nspec_dim"]],
        ]
        self.decode_layers: list[
            TypedLayer[
                ks.layers.Dense,
                FTensor[S["batch_dim nspec_dim _"]],
                FTensor[S["batch_dim nspec_dim decode_dim"]],
            ]
        ]
        self.batch_normalisation_layers: list[
            TypedLayer[
                ks.layers.BatchNormalization | ks.layers.Identity,
                FTensor[S["batch_dim nspec_dim decode_dim"]],
                FTensor[S["batch_dim nspec_dim decode_dim"]],
            ]
        ]
        self.decode_output_layer: TypedLayer[
            ks.layers.Dense,
            FTensor[S["batch_dim nspec_dim decode_dim"]],
            FTensor[S["batch_dim nspec_dim wl_dim"]],
        ]
        self.colourlaw_layer: TypedLayer[
            ks.layers.Dense | ks.layers.Identity,
            FTensor[S["batch_dim nspec_dim 1"]],
            FTensor[S["batch_dim nspec_dim wl_dim"]],
        ]

    @override
    def build(self, input_shape: "DecoderInputsShape") -> None:
        (_batch_dim, nspec_dim, _n_latents), (_batch_dim, _nspec_dim, _phase_dim) = (
            input_shape
        )
        # Project from input dimensions into nspec_dim dimensions
        self.decode_nspec_layer = TypedLayer(
            ks.layers.Dense(
                nspec_dim,
                activation=self.activation,
                kernel_regularizer=self.regulariser,
            )
        )

        # Decode from nspec_dim dimensions into intermediate dimensions
        self.decode_layers = []
        self.batch_normalisation_layers = []
        for decode_dim in self.decode_dims:
            self.decode_layers.append(
                TypedLayer(
                    ks.layers.Dense(
                        decode_dim,
                        activation=self.activation,
                        kernel_regularizer=self.regulariser,
                    )
                )
            )

            # Batch normalisation layer
            self.batch_normalisation_layers.append(
                TypedLayer(
                    ks.layers.BatchNormalization()
                    if self.batch_normalisation
                    else ks.layers.Identity()
                )
            )

        # Decode from intermediate dimensions to output dimensions
        self.decode_output_layer = TypedLayer(
            ks.layers.Dense(self.wl_dim, kernel_regularizer=self.regulariser)
        )

        # Colourlaw
        self.colourlaw_layer = TypedLayer(
            ks.layers.Dense(
                self.wl_dim,
                kernel_initializer=None
                if self.colourlaw is None
                else tf.constant_initializer(self.colourlaw),
                use_bias=False,
                trainable=True,
                kernel_constraint=None
                if self.colourlaw is None
                else ks.constraints.NonNeg(),
            )
            if self.n_physical > 0
            else ks.layers.Identity()
        )

    @override
    def call(
        self,
        inputs: "DecoderInputs",
        mask: "ITensor[S['batch_dim nspec_dim wl_dim']] | None" = None,
        training: bool | None = None,
    ) -> "DecoderOutputs":
        training = False if training is None else training

        input_latents = inputs[0]
        input_phase = inputs[1]
        input_mask = (
            mask
            if mask is not None
            else tf.ones((tf.shape(input_phase)[:-1], self.wl_dim), dtype=tf.int32)
        )

        # Extract physical parameters (if applicable)
        if self.n_physical > 0:
            delta_av_latent = input_latents[:, :, 0:1]
            zs_latent = input_latents[:, :, 1 : self.n_zs + 1]
            delta_m_latent = input_latents[:, :, self.n_zs + 1 : self.n_zs + 2]
            delta_p_latent = input_latents[:, :, self.n_zs + 2 : self.n_zs + 3]
        else:
            delta_av_latent = tf.expand_dims(tf.zeros_like(input_phase), axis=-1)
            zs_latent = input_latents[:, :, 0:]
            delta_m_latent = tf.expand_dims(tf.zeros_like(input_phase), axis=-1)
            delta_p_latent = tf.expand_dims(tf.zeros_like(input_phase), axis=-1)
        if TYPE_CHECKING:
            delta_av_latent = cast(
                "FTensor[S['batch_dim nspec_dim 1']]", delta_av_latent
            )
            zs_latent = cast("FTensor[S['batch_dim nspec_dim n_zs']]", zs_latent)
            delta_m_latent = cast("FTensor[S['batch_dim nspec_dim 1']]", delta_m_latent)
            delta_p_latent = cast("FTensor[S['batch_dim nspec_dim 1']]", delta_p_latent)

        # Apply Δ𝓅 shift
        input_phase += delta_p_latent

        # Create initial input layer
        x: FTensor[S["batch_dim nspec_dim n_zs+phase_dim"]] = ks.layers.concatenate([
            zs_latent,
            input_phase,
        ])

        # Decode from input (latent) dimensions to nspec_dim dimensions
        x = self.decode_nspec_layer(x)

        # Decode from nspec_dim dimensions to intermediate dimensions
        for i, decode_layer in enumerate(self.decode_layers):
            x = decode_layer(x)
            x = self.batch_normalisation_layers[i](x)

        # Decode from intermediate dimensions to output dimension
        amplitude = self.decode_output_layer(x)

        # Calculate Colourlaw
        colourlaw = self.colourlaw_layer(delta_av_latent)

        # Apply ΔAᵥ / Δℳ  shift
        if self.n_physical > 0:
            amplitude *= tf.pow(10.0, -0.4 * colourlaw * delta_m_latent)

        # Apply RELU activation function
        if not training:
            amplitude = tf.nn.relu(amplitude)

        # Zero out masked elements
        return amplitude * tf.cast(
            tf.reduce_max(input_mask, axis=-1, keepdims=True), tf.float32
        )


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEModel(ks.Model):
    def __init__(
        self,
        config: "PAEModel[TFPAEModel]",
        **kwargs: "Any",
    ) -> None:
        super().__init__(name=f"{config.name.split()[-1]}PAEModel", **kwargs)
        # --- Config ---
        options = cast("TFPAEModelConfig", config.options)
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force

        # --- Latent Dimensions ---
        self.n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents
        self.n_latents: int = self.n_physical + self.n_zs

        # --- Layers ---
        self.encoder: TFPAEEncoder = TFPAEEncoder(options, config.name)
        self.decoder: TFPAEDecoder = TFPAEDecoder(options, config.name)
        self.decoder.wl_dim = config.wl_dim

        # --- Training ---
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

        # Data Offsets
        self.phase_offset_scale: Literal[0, -1] | float = options.phase_offset_scale
        self.amplitude_offset_scale: float = options.amplitude_offset_scale
        self.mask_fraction: float = options.mask_fraction

        self._scheduler: type[ks.optimizers.schedules.LearningRateSchedule] = (
            options.scheduler_cls
        )
        self._optimiser: type[ks.optimizers.Optimizer] = options.optimiser_cls
        self._loss: ks.losses.Loss = options.loss_cls()

        self.stage: Stage
        self._epoch: int = 0

        # --- Metrics ---
        self.loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss")
        self.mae_metric: ks.metrics.Metric = ks.metrics.MeanAbsoluteError(name="mae")

    @property
    @override
    def metrics(self) -> list[ks.metrics.Metric]:
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mae_metric]

    @override
    def call(
        self,
        inputs: "EncoderInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "DecoderOutputs":
        training = False if training is None else training

        input_phase = inputs[0]
        encoded = self.encoder(inputs, training=training, mask=mask)
        if TYPE_CHECKING:
            encoded = cast("FTensor[S['batch_dim nspec_dim n_latents']]", encoded)

        decoded = self.decoder((encoded, input_phase), training=training, mask=mask)
        if TYPE_CHECKING:
            decoded = cast("FTensor[S['batch_dim nspec_dim wl_dim']]", decoded)

        return decoded

    @override
    def compute_loss(
        self,
        x: "TensorCompatible | None" = None,
        y: "TensorCompatible | None" = None,
        y_pred: "TensorCompatible | None" = None,
        sample_weight: "Any | None" = None,
        training: bool | None = None,
    ) -> "FTensor[S['']] | None":
        training = False if training is None else training
        if y is None or y_pred is None:
            return None
        loss = self._loss(y_true=y, y_pred=y_pred)
        if TYPE_CHECKING:
            loss = cast("FTensor[S['']]", loss)
        overall_loss = loss
        recon_loss = loss
        covariant_loss = loss
        # TODO: Other loss terms

        return loss

    @override
    def train_step(
        self,
        data: "TensorCompatible",
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:
        training = True

        # === Per Epoch Setup ===
        # Fixed RNG
        tf.random.set_seed(self._epoch)
        self._epoch += 1

        # --- Setup Data ---
        (phase, _d_phase, amplitude, _d_amplitude, mask) = self.prep_data_per_epoch(
            cast("EpochInputs", data)
        )

        with tf.GradientTape() as tape:
            output_amp = self((phase, amplitude), training=training, mask=mask)
            if TYPE_CHECKING:
                output_amp = cast(
                    "FTensor[S['batch_dim nspec_dim wl_dim']]", output_amp
                )
            loss = self.compute_loss(
                x=phase, y=amplitude, y_pred=output_amp, training=training
            )
        if loss is None:
            return {m.name: m.result() for m in self.metrics}

        gradients = tape.gradient(loss, self.trainable_variables)
        cast("ks.optimizers.Optimizer", self.optimizer).apply_gradients(
            zip(gradients, self.trainable_variables, strict=True)
        )

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(amplitude, output_amp)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train_model(
        self,
        stage: "Stage",
    ) -> ks.callbacks.History:
        self.stage = stage

        # === Setup Encoder ===
        self.encoder.stage_num = self.stage.stage
        self.encoder.moving_means = tf.convert_to_tensor(self.stage.moving_means)

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        if self.stage.savepath is not None:
            # --- Backup & Restore ---
            # Backup checkpoints each epoch and restore if training got cancelled midway through
            if not self.force:
                backup_dir = self.stage.savepath / "backups"
                backup_callback = ks.callbacks.BackupAndRestore(str(backup_dir))
                callbacks.append(backup_callback)

            # --- Model Checkpoint ---
            checkpoint_callback = ks.callbacks.ModelCheckpoint(
                str(self.stage.savepath / self.weights_path),
                save_best_only=self.save_best,
                save_weights_only=True,
            )
            callbacks.append(checkpoint_callback)

        # --- Terminate on NaN ---
        # Terminate training when a NaN loss is encountered
        callbacks.append(ks.callbacks.TerminateOnNaN())

        # === Setup Training ===
        schedule = self._scheduler(
            initial_learning_rate=self.stage.learning_rate,
            decay_steps=self.stage.learning_rate_decay_steps,
            decay_rate=self.stage.learning_rate_decay_rate,
        )
        optimiser = self._optimiser(
            learning_rate=schedule,
            weight_decay=self.stage.learning_rate_weight_decay_rate,
        )
        loss = self._loss
        self.compile(
            optimizer=optimiser,
            loss=loss,
            run_eagerly=self.stage.debug,
        )

        # === Build ===
        self(
            (self.stage.train_data.phase, self.stage.train_data.amplitude),
            training=False,
            mask=self.stage.train_data.mask,
        )

        self.log.debug("Trainable variables:")
        for var in self.trainable_variables:
            self.log.debug(f"{var.name}: {var.shape}")
        self.summary(print_fn=self.log.debug)  # Will show number of parameters

        if stage.loadpath is not None:
            self.load_checkpoint(stage.loadpath)

        # === Prep Data ===
        data = (
            self.stage.train_data.phase,
            self.stage.train_data.dphase,
            self.stage.train_data.amplitude,
            self.stage.train_data.sigma,
            self.stage.train_data.mask,
        )
        prep = self.prep_data(data)

        # === Train ===
        self._epoch = 0
        return self.fit(
            x=data,
            y=prep,
            initial_epoch=self._epoch,
            epochs=self.stage.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            # verbose=0,
        )

    def save_checkpoint(self) -> None:
        # Normalise mean of physical latents to 0 across all batches
        if self.n_physical > 0:
            phase = tf.convert_to_tensor(self.stage.train_data.phase)
            amplitude = tf.convert_to_tensor(self.stage.train_data.amplitude)
            mask = tf.convert_to_tensor(self.stage.train_data.mask)
            encoded = self.encoder((phase, amplitude), training=False, mask=mask)
            if TYPE_CHECKING:
                encoded = cast("FTensor[S['n_sn nspec_dim n_latents']]", encoded)

            latents_sum = tf.reduce_sum(tf.reduce_sum(encoded, axis=0), axis=0)
            latents_num = tf.reduce_sum(
                tf.reduce_sum(tf.ones_like(encoded), axis=0), axis=0
            )
            moving_means = latents_sum / latents_num
            if TYPE_CHECKING:
                latents_sum = cast("FTensor[S['n_latents']]", latents_sum)
                latents_num = cast("FTensor[S['n_latents']]", latents_num)
                moving_means = cast("FTensor[S['n_latents']]", moving_means)
            self.encoder.moving_means = moving_means
        if self.stage.savepath is not None:
            self.save_weights(self.stage.savepath / self.weights_path)
            self.save(self.stage.savepath / self.model_path)

    def load_checkpoint(self, loadpath: "Path") -> None:
        init_weights = self.encoder.encode_output_layer.get_weights()[0]
        self.load_weights(loadpath / self.weights_path)
        weights = self.encoder.encode_output_layer.get_weights()[0]

        # Set the weights of the newly introduced latent parameter to effectively 0
        #   Since the initial weights are random values which we then divide by 100
        weights[:, self.stage.stage - 1] = init_weights[:, self.stage.stage - 1] / 100
        self.encoder.encode_output_layer.set_weights([weights])

    def prep_data(self, data: "RawData") -> "PrepData":
        (_phase, _d_phase, _amplitude, _d_amplitude, mask) = data
        n_sn, _nspec_dim, _wl_dim = tf.shape(mask)

        # === Randomised Spectral Masking ===
        # We want a subset of each SN's spectra to be randomly masked
        # We do this by first building a mask tensor, masking the first n spectra per SN
        # Where n is the number of spectra we want to mask for that SNe

        # We then shuffle this mask to randomise which spectra make up this subset.

        # --- Build Masking tensor ---

        # Which spectra are currently masked for each SN
        spec_mask = tf.reduce_max(mask, axis=-1)
        if TYPE_CHECKING:
            spec_mask = cast("ITensor[S['n_sn nspec_dim']]", spec_mask)

        # The number of unmasked spectra for each SN
        n_unmasked_spectra_per_sn = tf.reduce_sum(spec_mask, axis=-1)
        if TYPE_CHECKING:
            n_unmasked_spectra_per_sn = cast(
                "ITensor[S['n_sn']]", n_unmasked_spectra_per_sn
            )

        # Determine how many spectra to mask for each SN
        #   by multiplying the number of unmasked SN by the mask_fraction
        n_spectra_to_mask_per_sn = tf.cast(
            tf.round(
                self.mask_fraction * tf.cast(n_unmasked_spectra_per_sn, tf.float32)
            ),
            tf.int32,
        )
        if TYPE_CHECKING:
            n_spectra_to_mask_per_sn = cast(
                "ITensor[S['n_sn']]", n_spectra_to_mask_per_sn
            )

        # How many spectra we are going to mask in total

        # Get the indices (batch_ind, nspec_ind) of each unmasked spectrum
        unmasked_inds = tf.cast(tf.where(tf.reduce_max(mask, axis=-1)), tf.int32)
        if TYPE_CHECKING:
            unmasked_inds = cast("ITensor[S['n_unmasked 2']]", unmasked_inds)

        # For each unmasked spectra, get the index of its corresponding SN
        unmasked_sn_inds = unmasked_inds[:, 0]
        if TYPE_CHECKING:
            unmasked_sn_inds = cast("ITensor[S['n_unmasked']]", unmasked_sn_inds)

        # For each unmasked spectra, get its index
        unmasked_spec_inds = unmasked_inds[:, 1]
        if TYPE_CHECKING:
            unmasked_spec_inds = cast("ITensor[S['n_unmasked']]", unmasked_spec_inds)

        # For each unmasked spectra, get the number of spectra we want to mask in its corresponding SN
        n_spectra_to_mask_per_spectra = tf.cast(
            tf.gather(n_spectra_to_mask_per_sn, unmasked_sn_inds), tf.int32
        )
        if TYPE_CHECKING:
            n_spectra_to_mask_per_spectra = cast(
                "ITensor[S['n_unmasked']]", n_spectra_to_mask_per_spectra
            )

        # Determine which spectrum should be masked.
        #   If a spectrum's index is less than the number of spectra to mask for that SN, then mask that spectrum.
        #   This essentially means we are masking the first `n_spectra_to_mask` spectra of each SN.
        to_mask = unmasked_spec_inds <= n_spectra_to_mask_per_spectra
        if TYPE_CHECKING:
            to_mask = cast("ITensor[S['n_unmasked']]", to_mask)

        # Get the indices of each spectrum we are going to mask.
        inds_to_mask = tf.boolean_mask(unmasked_inds, to_mask)
        if TYPE_CHECKING:
            inds_to_mask = cast("ITensor[S['n_to_mask 2']]", inds_to_mask)

        # Since we want to mask these spectra, we will update the mask at these indices with 0
        #   So this tensor just contains the value (0) we want to update the mask with
        zeros_mask = tf.zeros(tf.shape(inds_to_mask)[0], dtype=tf.int32)
        if TYPE_CHECKING:
            zeros_mask = cast("ITensor[S['n_unmasked']]", zeros_mask)

        # The masking tensor we are going to apply these updates to
        ones_mask = tf.ones_like(spec_mask, dtype=tf.int32)
        if TYPE_CHECKING:
            ones_mask = cast("ITensor[S['n_sn nspec_dim']]", ones_mask)

        # Update the values in `ones_mask` at the indices in `inds_to_mask` with the corresponding values in `zeros_mask`.
        unshuffled_mask = tf.tensor_scatter_nd_update(
            ones_mask, inds_to_mask, zeros_mask
        )
        if TYPE_CHECKING:
            unshuffled_mask = cast("ITensor[S['n_sn nspec_dim']]", unshuffled_mask)

        # --- Shuffle Masking Tensor ---

        # For each unmasked spectra, get the number of spectra we want to shuffle (i.e. the original number of unmasked spectra) in its corresponding SN
        n_spectra_to_shuffle_per_spectra = tf.cast(
            tf.gather(n_unmasked_spectra_per_sn, unmasked_sn_inds), tf.int32
        )
        if TYPE_CHECKING:
            n_spectra_to_shuffle_per_spectra = cast(
                "ITensor[S['n_unmasked']]", n_spectra_to_shuffle_per_spectra
            )

        # Determine which spectrum should be shuffled.
        #   If a spectrum's index is less than the number of spectra to shuffle for that SN, then shuffle that spectrum.
        #   This essentially means we are shuffling the first `n_spectra_to_shuffle` spectra of each SN.
        to_shuffle = unmasked_spec_inds <= n_spectra_to_shuffle_per_spectra
        if TYPE_CHECKING:
            to_shuffle = cast("ITensor[S['n_unmasked']]", to_shuffle)

        # Get the indices of each spectrum we are going to shuffle.
        inds_to_shuffle = tf.boolean_mask(unmasked_inds, to_shuffle)
        if TYPE_CHECKING:
            inds_to_shuffle = cast("ITensor[S['n_to_shuffle 2']]", inds_to_shuffle)

        # For each spectrum we will shuffle, get the index of its corresponding SN
        sn_inds = inds_to_shuffle[:, 0]
        if TYPE_CHECKING:
            sn_inds = cast("ITensor[S['n_to_shuffle']]", sn_inds)

        # For each spectrum we will shuffle, get its index
        spec_inds_to_shuffle = inds_to_shuffle[:, 1]
        if TYPE_CHECKING:
            spec_inds_to_shuffle = cast(
                "ITensor[S['n_to_shuffle']]", spec_inds_to_shuffle
            )

        # For each SN, get the index of each spectra we want to shuffle.
        #   Since different SNe will have a different number of spectra to shuffle, this is a ragged tensor
        spec_inds_to_shuffle_per_sn = tf.RaggedTensor.from_value_rowids(
            spec_inds_to_shuffle,
            sn_inds,
        )
        if TYPE_CHECKING:
            spec_inds_to_shuffle_per_sn = cast(
                "IRTensor[S['n_sn n_spec_to_shuffle(sn)']]", spec_inds_to_shuffle_per_sn
            )

        # For each SN, construct a tensor of the (sn_ind, nspec_ind) indices of each spectrum we are going to shuffle
        # This is basically splitting spec_inds_to_shuffle into n_sn tensors, each of length n_spec_to_shuffle(sn)
        # We need to do this so that these indices can be batched properly
        shuffle_inds = tf.ragged.constant([
            p.numpy()
            for p in tf.dynamic_partition(
                spec_inds_to_shuffle, sn_inds, num_partitions=n_sn
            )
        ])
        if TYPE_CHECKING:
            shuffle_inds = cast(
                "IRTensor[S['n_sn n_spec_to_shuffle(sn)']]", shuffle_inds
            )

        return (
            unshuffled_mask,
            shuffle_inds,
            spec_inds_to_shuffle_per_sn,
        )

    @tf.function
    def prep_data_per_epoch(self, data: "EpochInputs") -> "ModelInputs":
        (
            (phase, d_phase, amplitude, d_amplitude, mask),
            (
                unshuffled_mask,
                shuffle_inds,
                spec_inds_to_shuffle_per_sn,
            ),
        ) = data

        # === Randomised Data Offsets ===
        # Every epoch we want to randomly shift some of the data as a countermeasure to overfitting

        # --- Phase Offset ---
        if self.phase_offset_scale != 0:
            d_phase_shape = tf.shape(d_phase)
            if self.phase_offset_scale == 0:
                phase_offset = tf.zeros_like(d_phase)
            elif self.phase_offset_scale == -1:
                phase_offset = d_phase * tf.random.normal(d_phase_shape)
            else:
                phase_offset = (
                    tf.ones_like(d_phase)
                    * self.phase_offset_scale
                    * tf.random.normal(d_phase_shape)
                )
            if TYPE_CHECKING:
                phase_offset = cast(
                    "FTensor[S['batch_dim nspec_dim phase_dim']]", phase_offset
                )

            phase += phase_offset
            if TYPE_CHECKING:
                phase = cast("FTensor[S['batch_dim nspec_dim phase_dim']]", phase)

        # --- Amplitude Offset ---
        if self.amplitude_offset_scale != 0:
            amplitude_offset = (
                d_amplitude
                * self.amplitude_offset_scale
                * tf.random.normal(tf.shape(d_amplitude))
            )
            if TYPE_CHECKING:
                amplitude_offset = cast(
                    "FTensor[S['batch_dim nspec_dim wl_dim']]", amplitude_offset
                )

            amplitude += amplitude_offset
            if TYPE_CHECKING:
                amplitude = cast("FTensor[S['batch_dim nspec_dim wl_dim']]", amplitude)

        # --- Spectral Masking ---
        if self.mask_fraction != 0:
            # For each SN, shuffle the indices of each spectra we want to shuffle
            shuffled_inds = tf.ragged.map_flat_values(
                tf.random.shuffle, spec_inds_to_shuffle_per_sn
            )
            if TYPE_CHECKING:
                shuffled_inds = cast(
                    "IRTensor[Literal['batch_dim n_spec_to_shuffle(sn)']]",
                    shuffled_inds,
                )

            # Now that we've shuffled the indices of each spectra for each SN,
            #   We need to shuffle the actual spectra of each SN.
            #   Since the spectra were only shuffled with others from the same SN,
            #   the SN index associated with each shuffled spectra is the same too
            #   So this just repeats the SN index once for each shuffled spectra associated with that SN.
            sn_inds_per_spec = tf.repeat(
                tf.range(shuffled_inds.nrows()), shuffled_inds.row_lengths()
            )
            if TYPE_CHECKING:
                sn_inds_per_spec = cast(
                    "IRTensor[S['n_sn n_spec_to_shuffle(sn)']]", sn_inds_per_spec
                )

            # Flatten out the ragged shuffled_indices tensor, and stack with the gather_sn_indices tensor.
            # This tensor encodes where each spectrum will be shuffled to.
            orig_inds_to_shuffled_inds = tf.stack(
                (sn_inds_per_spec, shuffled_inds.flat_values), axis=1
            )
            if TYPE_CHECKING:
                orig_inds_to_shuffled_inds = cast(
                    "ITensor[S['n_spec_to_shuffle 2']]", orig_inds_to_shuffled_inds
                )

            # For each shuffled index in `orig_inds_to_shuffled_inds`, get the corresponding element in `unshuffled_mask`
            # This allows us to randomise which spectra are masked, rather than always masking the first `n_spectra_to_mask` spectra
            spec_masks = tf.gather_nd(unshuffled_mask, orig_inds_to_shuffled_inds)
            if TYPE_CHECKING:
                spec_masks = cast("ITensor[S['n_spec_to_shuffle']]", spec_masks)

            # The original indices we are shuffling
            sn_inds = tf.cast(
                tf.repeat(tf.range(shuffle_inds.nrows()), shuffle_inds.row_lengths()),
                tf.int32,
            )
            spec_inds_to_shuffle = tf.stack(
                [sn_inds, shuffled_inds.flat_values], axis=1
            )
            if TYPE_CHECKING:
                spec_inds_to_shuffle = cast(
                    "ITensor[S['n_spec_to_shuffle 2']]", spec_inds_to_shuffle
                )
            # The final mask which randomly masks a fraction of spectra per SN
            shuffled_mask = tf.tensor_scatter_nd_update(
                unshuffled_mask, spec_inds_to_shuffle, spec_masks
            )
            if TYPE_CHECKING:
                shuffled_mask = cast("ITensor[S['batch_dim nspec_dim']]", shuffled_mask)

            # Mask every amplitude for any spectra which has been randomly masked
            shuffled_amp_mask = tf.tile(
                tf.expand_dims(shuffled_mask, axis=-1), [1, 1, tf.shape(mask)[-1]]
            )
            if TYPE_CHECKING:
                shuffled_amp_mask = cast(
                    "ITensor[S['batch_dim nspec_dim wl_dim']]", shuffled_amp_mask
                )
            mask *= shuffled_amp_mask

        return (phase, d_phase, amplitude, d_amplitude, mask)
