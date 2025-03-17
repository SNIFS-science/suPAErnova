from typing import TYPE_CHECKING, override

import keras
import tensorflow as tf
from tensorflow.keras import layers

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorflow import keras as ks

    from suPAErnova.utils.suPAErnova_types import CONFIG


@keras.saving.register_keras_serializable()
class FunctionLayer(layers.Layer):
    _function_registry: "CONFIG[Callable]" = {}

    def __init__(self, fn_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn_name = fn_name

    def fn(self):
        fn = self._function_registry.get(self.fn_name)
        if fn is None:
            err = f"Unknown function '{self.fn_name}' during deserialisation"
            raise ValueError(err)
        return fn

    @override
    def call(self, x: "ks.KerasTensor", **kwargs):
        if isinstance(x, tf.SparseTensor):
            # Convert sparse tensor to dense tensor
            x = tf.sparse.to_dense(x)
        return self.fn()(x, **kwargs)

    @override
    def get_config(self):
        config = super().get_config()
        config["fn_name"] = self.fn_name
        return config

    @override
    @classmethod
    def from_config(cls, config):
        fn_name = config.pop("fn_name")
        return cls(fn_name, **config)

    @classmethod
    def register_function(cls, fn_name: str, fn: "Callable") -> None:
        cls._function_registry[fn_name] = fn


FunctionLayer.register_function("reduce_max", tf.reduce_max)
reduce_max = FunctionLayer("reduce_max")

FunctionLayer.register_function("reduce_min", tf.reduce_min)
reduce_min = FunctionLayer("reduce_min")

FunctionLayer.register_function("reduce_sum", tf.reduce_sum)
reduce_sum = FunctionLayer("reduce_sum")

FunctionLayer.register_function("maximum", tf.maximum)
maximum = FunctionLayer("maximum")

FunctionLayer.register_function("relu", tf.nn.relu)
relu = FunctionLayer("relu")
