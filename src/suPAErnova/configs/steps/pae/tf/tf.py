from typing import override
from functools import cached_property
from collections.abc import Callable

from pydantic import PositiveFloat, computed_field
import tensorflow as tf
from tensorflow import keras as ks

from suPAErnova.configs.steps import ConfigInputObject, validate_object
from suPAErnova.configs.steps.pae.model import PAEModelConfig

ActivationObject = Callable[[tf.Tensor], tf.Tensor]
RegulariserObject = Callable[[tf.Tensor], tf.Tensor] | type[ks.regularizers.Regularizer]
OptimiserObject = type[ks.optimizers.Optimizer]
LossObject = Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | type[ks.losses.Loss]


def validate_activation(activation: ConfigInputObject[ActivationObject]):
    return validate_object(
        activation, dummy_obj=ks.activations.linear, mod=ks.activations
    )


def validate_kernel_regulariser(
    kernel_regulariser: ConfigInputObject[RegulariserObject],
):
    return validate_object(
        kernel_regulariser, dummy_obj=ks.regularizers.Regularizer, mod=ks.regularizers
    )


def validate_optimiser(
    optimiser: ConfigInputObject[OptimiserObject],
):
    return validate_object(
        optimiser, dummy_obj=ks.optimizers.Optimizer, mod=ks.optimizers
    )


def validate_loss(
    loss: ConfigInputObject[LossObject],
):
    try:
        return validate_object(loss, dummy_obj=ks.losses.Loss, mod=ks.losses)
    except ValueError:
        return validate_object(loss, dummy_obj=ks.losses.mae, mod=ks.losses)


class TFPAEModelConfig(PAEModelConfig):
    # --- Training ---
    activation: ConfigInputObject[ActivationObject]

    @computed_field
    @cached_property
    def activation_fn(self) -> ActivationObject:
        return validate_activation(self.activation)

    kernel_regulariser: ConfigInputObject[RegulariserObject]

    @computed_field
    @cached_property
    def kernel_regulariser_cls(self) -> RegulariserObject:
        regulariser = validate_kernel_regulariser(self.kernel_regulariser)
        if isinstance(regulariser, type):
            return regulariser

        class CustomRegulariser(ks.regularizers.Regularizer):
            @override
            def __call__(self, x: tf.Tensor) -> tf.Tensor:
                return regulariser(x)

        return CustomRegulariser

    kernel_regulariser_penalty: PositiveFloat

    optimiser: ConfigInputObject[OptimiserObject]

    @computed_field
    @cached_property
    def optimiser_cls(self) -> OptimiserObject:
        return validate_optimiser(self.optimiser)

    loss: ConfigInputObject[LossObject]

    @computed_field
    @cached_property
    def loss_fn(self) -> type[ks.losses.Loss]:
        loss = validate_loss(self.loss)
        if isinstance(loss, type):
            return loss

        class CustomLoss(ks.losses.Loss):
            @override
            def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                return loss(y_true, y_pred)

        return CustomLoss
