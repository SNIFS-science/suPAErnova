from typing import TYPE_CHECKING, override

from keras import layers
import tensorflow as tf

if TYPE_CHECKING:
    from collections.abc import Callable

    import keras as ks


class FunctionLayer(layers.Layer):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    @override
    def call(self, x: "ks.KerasTensor", **kwargs):
        if isinstance(x, tf.SparseTensor):
            # Convert sparse tensor to dense tensor
            x = tf.sparse.to_dense(x)
        return self.fn(x, **kwargs)


class ReduceMax(FunctionLayer):
    def __init__(self) -> None:
        super().__init__(tf.reduce_max)


reduce_max = ReduceMax()


class ReduceMin(FunctionLayer):
    def __init__(self) -> None:
        super().__init__(tf.reduce_min)


reduce_min = ReduceMin()


class ReduceSum(FunctionLayer):
    def __init__(self) -> None:
        super().__init__(tf.reduce_sum)


reduce_sum = ReduceSum()
