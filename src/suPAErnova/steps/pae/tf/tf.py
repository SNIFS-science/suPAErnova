# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, Any, Literal, Annotated, cast, override

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

if TYPE_CHECKING:
    from logging import Logger
    from collections.abc import Callable

    from suPAErnova.steps.pae.model import Stage, PAEModel
    from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig

    # Annotated Layers
    TensorShape = tuple[Any, ...]
    type Tensor[Shape: TensorShape] = Annotated[tf.Tensor, Shape]
    type Layer[
        L: ks.layers.Layer,
        InputShape: TensorShape,
        OutputShape: TensorShape,
    ] = Annotated[L, InputShape, OutputShape]

    # Inputs
    type InputPhaseShape[BatchDim, NSpecDim, PhaseDim] = tuple[
        BatchDim, NSpecDim, PhaseDim
    ]
    type InputPhase[BatchDim, NSpecDim, PhaseDim] = Tensor[
        InputPhaseShape[BatchDim, NSpecDim, PhaseDim]
    ]

    InputDPhaseShape = InputPhaseShape
    type InputDPhase[BatchDim, NSpecDim, PhaseDim] = Tensor[
        InputDPhaseShape[BatchDim, NSpecDim, PhaseDim]
    ]

    type InputAmpShape[BatchDim, NSpecDim, WLDim] = tuple[BatchDim, NSpecDim, WLDim]
    type InputAmp[BatchDim, NSpecDim, WLDim] = Tensor[
        InputAmpShape[BatchDim, NSpecDim, WLDim]
    ]

    InputDAmpShape = InputAmpShape
    type InputDAmp[BatchDim, NSpecDim, WLDim] = Tensor[
        InputDAmpShape[BatchDim, NSpecDim, WLDim]
    ]

    InputMaskShape = InputAmpShape
    type InputMask[BatchDim, NSpecDim, WLDim] = Tensor[
        InputMaskShape[BatchDim, NSpecDim, WLDim]
    ]

    # Latents
    type Latent[BatchDim, NSpecDim] = Tensor[tuple[BatchDim, NSpecDim, Literal[1]]]
    DeltaAvLatent = Latent
    DeltaMLatent = Latent
    DeltaPLatent = Latent
    ZLatent = Latent

    type PhysicalLatents[BatchDim, NSpecDim] = tuple[
        DeltaAvLatent[BatchDim, NSpecDim],
        DeltaMLatent[BatchDim, NSpecDim],
        DeltaPLatent[BatchDim, NSpecDim],
    ]
    type ZLatents[BatchDim, NSpecDim] = tuple[ZLatent[BatchDim, NSpecDim], ...]

    type LatentsShape[BatchDim, NSpecDim, NLatents] = tuple[
        BatchDim, NSpecDim, NLatents
    ]
    type Latents[BatchDim, NSpecDim, NLatents] = Tensor[
        LatentsShape[BatchDim, NSpecDim, NLatents]
    ]

    # Encoder
    type EncoderInputsShape[BatchDim, NSpecDim, PhaseDim, WLDim] = tuple[
        InputPhaseShape[BatchDim, NSpecDim, PhaseDim],
        InputAmpShape[BatchDim, NSpecDim, WLDim],
    ]
    type EncoderInputs[BatchDim, NSpecDim, PhaseDim, WLDim] = tuple[
        Tensor[InputPhaseShape[BatchDim, NSpecDim, PhaseDim]],
        Tensor[InputAmpShape[BatchDim, NSpecDim, WLDim]],
    ]

    EncoderOutputsShape = LatentsShape
    type EncoderOutputs[BatchDim, NSpecDim, NLatents] = Tensor[
        EncoderOutputsShape[BatchDim, NSpecDim, NLatents]
    ]

    # Decoder
    type DecoderInputsShape[BatchDim, NSpecDim, PhaseDim, NLatents] = tuple[
        EncoderOutputsShape[BatchDim, NSpecDim, NLatents],
        InputPhaseShape[BatchDim, NSpecDim, PhaseDim],
    ]
    type DecoderInputs[BatchDim, NSpecDim, PhaseDim, NLatents] = tuple[
        Tensor[EncoderOutputsShape[BatchDim, NSpecDim, NLatents]],
        Tensor[InputPhaseShape[BatchDim, NSpecDim, PhaseDim]],
    ]
    DecoderOutputsShape = InputAmpShape
    type DecoderOutputs[BatchDim, NSpecDim, WLDim] = Tensor[
        DecoderOutputsShape[BatchDim, NSpecDim, WLDim]
    ]

    # Model
    type ModelInputsShape[BatchDim, NSpecDim, PhaseDim, WLDim] = tuple[
        InputPhaseShape[BatchDim, NSpecDim, PhaseDim],
        InputDPhaseShape[BatchDim, NSpecDim, PhaseDim],
        InputAmpShape[BatchDim, NSpecDim, WLDim],
        InputDAmpShape[BatchDim, NSpecDim, WLDim],
        InputMaskShape[BatchDim, NSpecDim, WLDim],
    ]
    type ModelInputs[BatchDim, NSpecDim, PhaseDim, WLDim] = tuple[
        tuple[
            Tensor[InputPhaseShape[BatchDim, NSpecDim, PhaseDim]],
            Tensor[InputDPhaseShape[BatchDim, NSpecDim, PhaseDim]],
            Tensor[InputAmpShape[BatchDim, NSpecDim, WLDim]],
            Tensor[InputDAmpShape[BatchDim, NSpecDim, WLDim]],
            Tensor[InputMaskShape[BatchDim, NSpecDim, WLDim]],
        ]
    ]

    X = int

    from tensorflow._aliases import TensorCompatible


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEEncoder[
    BatchDim: int,
    NSpecDim: int,
    PhaseDim: int,
    WLDim: int,
    NLatents: int,
](ks.layers.Layer):
    def __init__(
        self,
        options: "TFPAEModelConfig",
        name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{name.split()[-1]}Encoder", **kwargs)

        # --- Config Params ---
        n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        n_zs: int = options.n_z_latents
        self.n_latents: int = n_physical + n_zs
        self.latents_z_mask: Tensor[tuple[NLatents]]
        self.latents_physical_mask: Tensor[tuple[NLatents]]

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
            Layer[
                ks.layers.Dense,
                tuple[BatchDim, WLDim, int],
                tuple[BatchDim, WLDim, int],
            ]
        ]

        self.dropout_layers: list[
            Layer[
                ks.layers.Dropout | ks.layers.Identity,
                tuple[BatchDim, WLDim, X],
                tuple[BatchDim, WLDim, X],
            ]
        ]
        self.batch_normalisation_layers: list[
            Layer[
                ks.layers.BatchNormalization | ks.layers.Identity,
                tuple[BatchDim, WLDim, X],
                tuple[BatchDim, WLDim, X],
            ]
        ]
        self.encode_nspec_layer: Layer[
            ks.layers.Dense,
            tuple[BatchDim, WLDim, X],
            tuple[BatchDim, WLDim, NSpecDim],
        ]

        self.encode_output_layer: Layer[
            ks.layers.Dense,
            tuple[BatchDim, WLDim, NSpecDim],
            tuple[BatchDim, WLDim, NLatents],
        ]
        self.repeat_latent_layer: Layer[
            ks.layers.RepeatVector,
            tuple[BatchDim, NLatents],
            tuple[BatchDim, NSpecDim, NLatents],
        ]

    @override
    def build(
        self, input_shape: "EncoderInputsShape[BatchDim, NSpecDim, PhaseDim, WLDim]"
    ) -> None:
        (_batch_dim, nspec_dim, _phase_dim), (_batch_dim, _nspec_dim, _wl_dim) = (
            input_shape
        )

        # Encode from input layer dimensions into intermediate dimensions
        self.encode_layers = []
        self.dropout_layers = []
        self.batch_normalisation_layers = []
        for n in self.encode_dims:
            self.encode_layers.append(
                ks.layers.Dense(
                    n,
                    activation=self.activation,
                    kernel_regularizer=self.regulariser,
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
            nspec_dim,
            activation=self.activation,
            kernel_regularizer=self.regulariser,
        )

        # Encode from nspec_dim dimensions into output (latent) dimensions
        self.encode_output_layer = ks.layers.Dense(
            self.n_latents,
            kernel_regularizer=self.regulariser,
            use_bias=False,
        )

        # Repeat latent vector to match nspec_dim
        self.repeat_latent_layer = ks.layers.RepeatVector(nspec_dim)

    @override
    def call(
        self,
        inputs: "EncoderInputs[BatchDim, NSpecDim, PhaseDim, WLDim]",
        mask: "InputMask[BatchDim, NSpecDim, WLDim] | None" = None,
        training: bool | None = None,
    ) -> "EncoderOutputs[BatchDim, NSpecDim, NLatents]":
        training = False if training is None else training

        input_phase = inputs[0]
        input_amp = inputs[1]
        input_mask = mask if mask is not None else tf.ones_like(input_amp)

        # Create initial input layer
        # (BatchDim, NSpecDim, WLDim + 1)
        x: Tensor[tuple[BatchDim, NSpecDim, int]] = ks.layers.concatenate([
            input_amp,
            input_phase,
        ])

        # Encode from input layers to intermediate dimensions
        # (BatchDim, NSpecDim, EncodeDim)
        for i, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x)
            x = self.dropout_layers[i](x, training=training)
            x = self.batch_normalisation_layers[i](x)

        # Encode from intermediate dimensions to nspec_dim dimensions
        # (BatchDim, NSpecDim, NSpecDim)
        x = self.encode_nspec_layer(x)
        if TYPE_CHECKING:
            x = cast("Tensor[tuple[BatchDim, NSpecDim, NSpecDim]]", x)

        # Encode from nspec_dim dimensions to output (latent) dimensions
        # (BatchDim, NSpecDim, NLatents)
        x = self.encode_output_layer(x)
        if TYPE_CHECKING:
            x = cast("Tensor[tuple[BatchDim, NSpecDim, NLatents]]", x)

        # Determine which spectra to keep
        # (BatchDim, NSpecDim, 1)
        is_kept = tf.cast(tf.reduce_min(input_mask, axis=-1, keepdims=True), tf.float32)
        if TYPE_CHECKING:
            is_kept = cast("Tensor[tuple[BatchDim, NSpecDim, Literal[1]]]", is_kept)

        # Latent tensor is the average of the latent values over all unmasked spectra
        # First sum latents over all unmasked spectra
        # (BatchDim, NLatents)
        batch_sum: Tensor[tuple[BatchDim, NLatents]] = tf.reduce_sum(
            x * is_kept, axis=-2
        )

        # Then determine the number of unmasked spectra
        # (BatchDim, 1)
        batch_num: Tensor[tuple[BatchDim, NLatents]] = tf.maximum(
            tf.reduce_sum(is_kept, axis=-2), y=1
        )

        # Finally, calculate the average
        # (BatchDim, NLatents)
        latents: Tensor[tuple[BatchDim, NLatents]] = batch_sum / batch_num

        if training:
            # Normalise the physical latents within this batch such that they have a mean of 0
            # (1, NLatents)
            latents_sum: Tensor[tuple[Literal[1], NLatents]] = tf.reduce_sum(
                latents, axis=0, keepdims=True
            )
            latents_num: Tensor[tuple[Literal[1], NLatents]] = tf.reduce_sum(
                tf.ones_like(latents), axis=0, keepdims=True
            )
            latents_mean: Tensor[tuple[Literal[1], NLatents]] = (
                self.latents_physical_mask * latents_sum / latents_num
            )
            latents -= latents_mean

        # Repeat latent layers across NSpecDim
        # (BatchDim, NSpecDim, NLatents)
        outputs: EncoderOutputs[BatchDim, NSpecDim, NLatents] = (
            self.repeat_latent_layer(latents)
        )

        return outputs


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEDecoder[
    BatchDim: int,
    NSpecDim: int,
    PhaseDim: int,
    WLDim: int,
    NLatents: int,
](ks.layers.Layer):
    def __init__(self, options: "TFPAEModelConfig", name: str, **kwargs: Any) -> None:
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
            colourlaw = tf.convert_to_tensor(colourlaw)
        self.colourlaw: tf.Tensor | None = colourlaw

        # --- Layers ---
        self.decode_nspec_layer: Layer[
            ks.layers.Dense,
            tuple[BatchDim, NSpecDim, int],
            tuple[BatchDim, NSpecDim, NSpecDim],
        ]
        self.decode_layers: list[
            Layer[
                ks.layers.Dense,
                tuple[BatchDim, NSpecDim, int],
                tuple[BatchDim, NSpecDim, int],
            ]
        ]
        self.batch_normalisation_layers: list[
            Layer[
                ks.layers.BatchNormalization | ks.layers.Identity,
                tuple[BatchDim, NSpecDim, X],
                tuple[BatchDim, NSpecDim, X],
            ]
        ]
        self.decode_output_layer: Layer[
            ks.layers.Dense,
            tuple[BatchDim, NSpecDim, int],
            tuple[BatchDim, NSpecDim, WLDim],
        ]
        self.colourlaw_layer: Layer[
            ks.layers.Dense | ks.layers.Identity,
            tuple[BatchDim, NSpecDim, Literal[1]],
            tuple[BatchDim, NSpecDim, WLDim],
        ]

    @override
    def build(
        self, input_shape: "DecoderInputsShape[BatchDim, NSpecDim, PhaseDim, NLatents]"
    ) -> None:
        (_batch_dim, nspec_dim, _n_latents), (_batch_dim, _nspec_dim, _phase_dim) = (
            input_shape
        )
        # Project from input dimensions into nspec_dim dimensions
        self.decode_nspec_layer = ks.layers.Dense(
            nspec_dim,
            activation=self.activation,
            kernel_regularizer=self.regulariser,
        )

        # Decode from nspec_dim dimensions into intermediate dimensions
        self.decode_layers = []
        self.batch_normalisation_layers = []
        for n in self.decode_dims:
            self.decode_layers.append(
                ks.layers.Dense(
                    n,
                    activation=self.activation,
                    kernel_regularizer=self.regulariser,
                )
            )

            # Batch normalisation layer
            if self.batch_normalisation:
                self.batch_normalisation_layers.append(ks.layers.BatchNormalization())
            else:
                self.batch_normalisation_layers.append(ks.layers.Identity())

        # Decode from intermediate dimensions to output dimensions
        self.decode_output_layer = ks.layers.Dense(
            self.wl_dim, kernel_regularizer=self.regulariser
        )

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
    def call(
        self,
        inputs: "DecoderInputs[BatchDim, NSpecDim, PhaseDim, NLatents]",
        mask: "InputMask[BatchDim, NSpecDim, WLDim] | None" = None,
        training: bool | None = None,
    ) -> "DecoderOutputs[BatchDim, NSpecDim, WLDim]":
        training = False if training is None else training

        input_latents = inputs[0]
        input_phase = inputs[1]

        input_mask = tf.cast(
            mask
            if mask is not None
            else tf.tile(
                tf.expand_dims(tf.ones_like(input_phase), axis=-1), [1, 1, self.wl_dim]
            ),
            tf.float32,
        )

        # Extract physical parameters (if applicable)
        if self.n_physical > 0:
            delta_av_latent = input_latents[..., 0:1]
            zs_latent = input_latents[..., 1 : self.n_zs + 1]
            delta_m_latent = input_latents[..., self.n_zs + 1 : self.n_zs + 2]
            delta_p_latent = input_latents[..., self.n_zs + 2 : self.n_zs + 3]
        else:
            delta_av_latent = tf.expand_dims(tf.zeros_like(input_phase), axis=-1)
            zs_latent = input_latents[..., 0:]
            delta_m_latent = tf.expand_dims(tf.zeros_like(input_phase), axis=-1)
            delta_p_latent = tf.expand_dims(tf.zeros_like(input_phase), axis=-1)

        # Apply Δ𝓅 shift
        input_phase += delta_p_latent

        # Create initial input layer
        # (BatchDim, NSpecDim, 1)
        x: Tensor[tuple[BatchDim, NSpecDim, Literal[1]]] = ks.layers.concatenate([
            zs_latent,
            input_phase,
        ])

        # Decode from input (latent) dimensions to nspec_dim dimensions
        # (BatchDim, NSpecDim, NSpecDim)
        x = self.decode_nspec_layer(x)
        if TYPE_CHECKING:
            x = cast("Tensor[tuple[BatchDim, NSpecDim, NSpecDim]]", x)

        # Decode from nspec_dim dimensions to intermediate dimensions
        # (BatchDim, NSpecDim, DecodeDim)
        for i, decode_layer in enumerate(self.decode_layers):
            x = decode_layer(x)
            x = self.batch_normalisation_layers[i](x)

        # Decode from intermediate dimensions to output dimension
        # (BatchDim, NSpecDim, WLDim)
        amplitude: Tensor[tuple[BatchDim, NSpecDim, WLDim]] = self.decode_output_layer(
            x
        )

        # Calculate Colourlaw
        # (BatchDim, NSpecDim, WLDim)
        colourlaw: Tensor[tuple[BatchDim, NSpecDim, WLDim]] = self.colourlaw_layer(
            delta_av_latent
        )

        # Apply ΔAᵥ / Δℳ  shift
        # (BatchDim, NSpecDim, WLDim)
        if self.n_physical > 0:
            amplitude *= tf.pow(10.0, -0.4 * colourlaw * delta_m_latent)

        # Apply RELU activation function
        # (BatchDim, NSpecDim, WLDim)
        if not training:
            amplitude = tf.nn.relu(amplitude)

        # Zero out masked elements
        # (BatchDim, NSpecDim, WLDim)
        outputs: DecoderOutputs[BatchDim, NSpecDim, WLDim] = amplitude * tf.reduce_max(
            input_mask, axis=-1, keepdims=True
        )
        return outputs


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEModel[
    BatchDim: int,
    NSpecDim: int,
    PhaseDim: int,
    WLDim: int,
    NLatents: int,
](ks.Model):
    def __init__(
        self,
        config: "PAEModel[TFPAEModel[int, int, int, int, int]]",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{config.name.split()[-1]}PAEModel", **kwargs)
        # --- Config ---
        options = cast("TFPAEModelConfig", config.options)
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose

        # --- Latent Dimensions ---
        self.n_physical: Literal[0, 3] = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents
        self.n_latents: int = self.n_physical + self.n_zs

        # --- Layers ---
        self.encoder: TFPAEEncoder[BatchDim, NSpecDim, PhaseDim, WLDim, NLatents] = (
            TFPAEEncoder(options, config.name)
        )

        self.decoder: TFPAEDecoder[BatchDim, NSpecDim, PhaseDim, WLDim, NLatents] = (
            TFPAEDecoder(options, config.name)
        )
        self.decoder.wl_dim = config.wl_dim

        # --- Training ---
        self.batch_size: int = config.batch_size

        # Data Offsets
        self.phase_offset_scale: Literal[0, -1] | float = config.phase_offset_scale
        self.amplitude_offset_scale: float = config.amplitude_offset_scale
        self.mask_fraction: float = config.mask_fraction

        self._scheduler: type[ks.optimizers.schedules.LearningRateSchedule] = (
            options.scheduler_cls
        )
        self._optimiser: type[ks.optimizers.Optimizer] = options.optimiser_cls
        self._loss: ks.losses.Loss = options.loss_cls()

        self.stage: Stage
        self._epoch: int

        # Pre-calculated tensors used per-epoch
        self.prep_mask_fraction: Tensor[tuple[BatchDim, NSpecDim, WLDim]]
        self.prep_shuffle_indices: Tensor[tuple[int, Literal[2]]]
        self.prep_shuffle_spectra_indices_per_sn: Tensor[tuple[BatchDim, int]]

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
        inputs: "EncoderInputs[BatchDim, NSpecDim, PhaseDim, WLDim]",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "DecoderOutputs[BatchDim, NSpecDim, WLDim]":
        training = False if training is None else training

        encoded: EncoderOutputs[BatchDim, NSpecDim, NLatents] = self.encoder(
            inputs, training=training, mask=mask
        )
        if training:
            # Mask latents which aren't being trained
            # The latents are ordered by training stage
            # ΔAᵥ -> zs -> Δℳ  -> Δ𝓅
            masked_latents = tf.zeros(self.n_latents - self.stage.stage)
            unmasked_latents = tf.ones(self.stage.stage)
            latent_mask = tf.concat((unmasked_latents, masked_latents), axis=0)
            encoded *= latent_mask
        else:
            # Normalise the physical latents within this batch such that the entire unbatched sample has a mean of 0
            encoded -= tf.convert_to_tensor(self.stage.moving_means)

        decoded: DecoderOutputs[BatchDim, NSpecDim, WLDim] = self.decoder(
            (encoded, inputs[0]), training=training, mask=mask
        )

        return decoded

    @override
    def compute_loss(
        self,
        x: "TensorCompatible | None" = None,
        y: "TensorCompatible | None" = None,
        y_pred: "TensorCompatible | None" = None,
        sample_weight: Any | None = None,
        training: bool | None = None,
    ) -> tf.Tensor | None:
        training = False if training is None else training
        if y is None or y_pred is None:
            return None
        loss = self._loss(y_true=y, y_pred=y_pred)
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
        # === Per Epoch Setup ===
        training = self.stage.training

        # Fixed RNG
        tf.random.set_seed(self._epoch)
        self._epoch += 1

        # --- Setup Data ---
        ((phase, _d_phase, amplitude, _d_amplitude, mask),) = self.prep_data_per_epoch(
            cast("ModelInputs[BatchDim, NSpecDim, PhaseDim, WLDim]", data)
        )

        with tf.GradientTape() as tape:
            output_amp: DecoderOutputs[BatchDim, NSpecDim, WLDim] = self(
                (phase, amplitude), training=training, mask=mask
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
    ) -> None:
        self.stage = stage

        # === Setup Training ===
        self._epoch = 0
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

        data = (
            (
                self.stage.train_data.phase,
                self.stage.train_data.dphase,
                self.stage.train_data.amplitude,
                self.stage.train_data.sigma,
                self.stage.train_data.mask,
            ),
        )

        # === Train ===
        self.fit(
            self.prep_data(
                cast("ModelInputs[BatchDim, NSpecDim, PhaseDim, WLDim]", data)
            )[0],
            initial_epoch=self._epoch,
            epochs=self.stage.epochs,
            batch_size=self.batch_size,
        )

    def prep_data(
        self, data: "ModelInputs[BatchDim, NSpecDim, PhaseDim, WLDim]"
    ) -> "ModelInputs[BatchDim, NSpecDim, PhaseDim, WLDim]":
        ((phase, d_phase, amplitude, d_amplitude, mask),) = data

        wl_dim = tf.shape(mask)[-1]

        # Mask Spectra
        # The goal here is to randomly mask a fraction of all valid (unmasked) spectra

        # --- Randomised Masking ---
        # Sum mask tensor over NSpecDim, gives the number of unmasked spectra for each SN
        # (BatchDim,)
        n_unmasked_spectra_per_sn: Tensor[tuple[BatchDim, WLDim]] = tf.reduce_sum(
            mask, axis=1
        )[:, 0]

        # Determine how many spectra to mask for each SN
        #   by multiplying the number of unmasked SN by the mask_fraction
        # (BatchDim,)
        n_spectra_to_mask_per_sn: Tensor[tuple[BatchDim]] = tf.cast(
            tf.round(
                self.mask_fraction * tf.cast(n_unmasked_spectra_per_sn, tf.float32)
            ),
            tf.int32,
        )

        # Get the indices (batch_ind, nspec_ind, wl_ind (unused)) of each unmasked spectrum
        # (n_unmasked_spectra, 3)
        unmasked_indices: Tensor[tuple[int, Literal[3]]] = tf.cast(
            tf.where(mask), tf.int32
        )

        # For each unmasked spectrum, get the index of its corresponding SN
        # (n_unmasked_spectra,)
        maskable_sn_indices: Tensor[tuple[int]] = unmasked_indices[:, 0]

        # For each unmasked spectrum, get its index
        # (n_unmasked_spectra,)
        maskable_spectra_indices: Tensor[tuple[int]] = unmasked_indices[:, 1]

        # Gather the number of spectra to mask per maskable sn
        # (n_unmasked_spectra,)
        n_spectra_to_mask_per_maskable_sn: Tensor[tuple[int]] = tf.cast(
            tf.gather(n_spectra_to_mask_per_sn, maskable_sn_indices), tf.int32
        )

        # Determine which spectrum should be masked.
        #   If a spectrum's index is less than the number of spectra to mask for that SN, then mask that spectrum.
        #   This essentially means we are masking the first `n_spectra_to_mask` spectra of each SN.
        # (n_unmasked_spectra,)
        maskable_indices: Tensor[tuple[int]] = (
            maskable_spectra_indices <= n_spectra_to_mask_per_maskable_sn
        )

        # Get the indices of each spectrum we are going to mask.
        # (n_spectra_to_mask, 3)
        mask_indices: Tensor[tuple[int, Literal[3]]] = tf.boolean_mask(
            unmasked_indices, maskable_indices
        )
        n_spectra_to_mask = tf.shape(mask_indices)[0]

        # Since we want to mask these spectra, we will update the mask at these indices with 0
        #   So this tensor just contains the value (0) we want to update the mask with
        # (n_spectra_to_mask,)
        mask_updates: Tensor[tuple[int]] = tf.zeros(n_spectra_to_mask, dtype=tf.int32)

        # The masking tensor we are going to apply these updates to
        # (BatchDim, NSpecDim, WLDim)
        base_mask: Tensor[tuple[BatchDim, NSpecDim, WLDim]] = tf.ones_like(
            mask, dtype=tf.int32
        )

        # Update the values in `base_mask` at the indices with `mask_indices` with the corresponding values in `mask_updates`.
        # (BatchDim, NSpecDim, WLDim)
        self.prep_mask_fraction = tf.tensor_scatter_nd_update(
            base_mask, mask_indices, mask_updates
        )

        # --- Randomised Shuffling ---

        # Gather the number of spectra to shuffle per maskable sn,
        # (n_unmasked_spectra,)
        n_spectra_to_shuffle_per_maskable_sn: Tensor[tuple[int]] = tf.cast(
            tf.gather(n_unmasked_spectra_per_sn, maskable_sn_indices), tf.int32
        )

        # Determine which spectrum should be shuffled.
        #   If a spectrum's index is less than the number of spectra to shuffle for that SN, then shuffle that spectrum.
        #   This essentially means we are shuffling the first `n_spectra_to_shuffle` spectra of each SN.
        # (n_unmasked_spectra,)
        shufflable_indices: Tensor[tuple[int]] = (
            maskable_spectra_indices <= n_spectra_to_shuffle_per_maskable_sn
        )

        # Get the indices of each spectrum we are going to shuffle.
        # (n_shuffle_spectra, 2)
        shuffle_indices: Tensor[tuple[int, Literal[2]]] = tf.boolean_mask(
            unmasked_indices, shufflable_indices
        )  # Get the (3 dimensional) indicies
        shuffle_indices = shuffle_indices[:, :-1]  # Only keep the first two indices
        stride_indices = tf.range(0, tf.shape(shuffle_indices)[0], wl_dim)
        self.prep_shuffle_indices = shuffle_indices = tf.gather(
            shuffle_indices, stride_indices
        )  # Only keep one index each WLDim step

        # For each spectrum we will shuffle, get the index of its corresponding SN
        # (n_shuffle_spectra,)
        shuffle_sn_indices: Tensor[tuple[int]] = shuffle_indices[:, 0]

        # For each spectrum we will shuffle, get its index
        # (n_shuffle_spectra,)
        shuffle_spectra_indices: Tensor[tuple[int]] = shuffle_indices[:, 1]

        # For each SN, get the index of each spectra we want to shuffle.
        #   Since different SNe will have a different number of spectra to shuffle, this is a ragged tensor
        # (BatchDim, n_spectra_to_shuffle_per_sn) where ∑n_spectra_to_shuffle_per_sn = n_shuffle_spectra
        self.prep_shuffle_spectra_indices_per_sn = tf.RaggedTensor.from_value_rowids(
            shuffle_spectra_indices,
            shuffle_sn_indices,
        )

        return ((phase, d_phase, amplitude, d_amplitude, mask),)

    def prep_data_per_epoch(
        self, data: "ModelInputs[BatchDim, NSpecDim, PhaseDim, WLDim]"
    ) -> "ModelInputs[BatchDim, NSpecDim, PhaseDim, WLDim]":
        ((phase, d_phase, amplitude, d_amplitude, mask),) = data

        # Precompute shapes
        shape_d_phase = tf.shape(d_phase)
        shape_d_amplitude = tf.shape(d_amplitude)

        # Phase Offset
        phase_offset: Tensor[tuple[BatchDim, NSpecDim, PhaseDim]]
        if self.phase_offset_scale == 0:
            phase_offset = tf.zeros_like(d_phase)
        elif self.phase_offset_scale == -1:
            phase_offset = d_phase * tf.random.normal(shape_d_phase)
        else:
            phase_offset = (
                tf.ones_like(d_phase)
                * self.phase_offset_scale
                * tf.random.normal(shape_d_phase)
            )
        phase += phase_offset

        # Amplitude Offset
        amplitude_offset: Tensor[tuple[BatchDim, NSpecDim, PhaseDim]] = (
            d_amplitude
            * self.amplitude_offset_scale
            * tf.random.normal(shape_d_amplitude)
        )
        amplitude += amplitude_offset

        ((phase, d_phase, amplitude, d_amplitude, mask),) = data
        # For each SN, shuffle the indices of each spectra we want to shuffle
        # (BatchDim, n_spectra_to_shuffle_per_sn) where ∑n_spectra_to_shuffle_per_sn = n_shuffle_spectra
        shuffled_indices: Tensor[tuple[BatchDim, int]] = tf.ragged.map_flat_values(
            tf.random.shuffle, self.prep_shuffle_spectra_indices_per_sn
        )

        # Now that we've shuffled the indices of each spectra for each SN,
        #   We need to shuffle the actual spectra of each SN.
        #   Since the spectra were only shuffled with others from the same SN,
        #   the SN index associated with each shuffled spectra is the same too
        #   So this just repeats the SN index once for each shuffled spectra associated with that SN.
        # (n_shuffle_spectra,)
        gather_sn_indices: Tensor[tuple[int]] = tf.repeat(
            tf.range(shuffled_indices.nrows()), shuffled_indices.row_lengths()
        )

        # Flatten out the ragged shuffled_indices tensor, and stack with the gather_sn_indices tensor.
        # This tensor encodes where each spectrum will be shuffled to.
        # (n_shuffle_spectra, 2)
        gather_shuffled_indices: Tensor[tuple[int, Literal[2]]] = tf.stack(
            (gather_sn_indices, shuffled_indices.flat_values), axis=1
        )

        # For each shuffled index in gather_shuffled_indices, get the corresponding element in `mask_fraction`
        # This allows us to randomise which spectra are masked, rather than always masking the first `n_spectra_to_mask` spectra
        # (n_shuffle_spectra, WLDim)
        shuffle_update: Tensor[tuple[int, WLDim]] = tf.gather_nd(
            self.prep_mask_fraction, gather_shuffled_indices
        )

        # The final mask which randomly masks a fraction of spectra per SN
        # (BatchDim, NSpecDim, WLDim)
        shuffle_and_mask: Tensor[tuple[BatchDim, NSpecDim, WLDim]] = (
            tf.tensor_scatter_nd_update(
                self.prep_mask_fraction, self.prep_shuffle_indices, shuffle_update
            )
        )

        mask *= shuffle_and_mask

        return ((phase, d_phase, amplitude, d_amplitude, mask),)
