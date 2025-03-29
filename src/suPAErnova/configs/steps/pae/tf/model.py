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
    def kernel_regulariser_cls(self) -> type[ks.regularizers.Regularizer]:
        regulariser = validate_kernel_regulariser(self.kernel_regulariser)
        if isinstance(regulariser, type):
            return regulariser

        class CustomRegulariser(ks.regularizers.Regularizer):
            @override
            def __call__(self, x: tf.Tensor) -> tf.Tensor:
                return regulariser(x)

        return CustomRegulariser

    kernel_regulariser_penalty: PositiveFloat
