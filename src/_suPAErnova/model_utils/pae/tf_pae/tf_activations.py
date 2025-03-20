from typing import TYPE_CHECKING

from tensorflow import keras as ks

if TYPE_CHECKING:
    from collections.abc import Callable

    import tensorflow as tf

    from suPAErnova.utils.suPAErnova_types import CONFIG

#   - "ELU": Use an [Exponential Linear Unit](https://www.tensorflow.org/api_docs/python/tf/keras/activations/elu) activation function
#   - "GELU": Use a [Gaussian Error Linear Unit](https://www.tensorflow.org/api_docs/python/tf/keras/activations/gelu) activation function
#   - "RELU": Use a [REctified Linear Unit](https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu) activation function
#   - "SWISH": Use a [Swish / Silu](https://www.tensorflow.org/api_docs/python/tf/keras/activations/silu) activation function
#   - "TANH": Use a [Hyperbolic Tangent](https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh) activation function
#   - "NULL": Custom activation function which returns None no matter what. Designed for use with custom user-defined activation functions, integrated through callbacks.


def null(_x: "tf.Tensor", _x_pred: "tf.Tensor", _kwargs: "CONFIG[tf.Tensor]") -> None:
    return None


activation: "CONFIG[Callable[[ks.activations._ActivationInput], tf.Tensor]]" = {
    "NULL": null,  # pyright:ignore[reportAssignmentType]
    "ELU": ks.activations.elu,
    "GELU": ks.activations.gelu,
    "RELU": ks.activations.relu,
    "SWISH": ks.activations.swish,
    "TANH": ks.activations.tanh,
}
