from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from .tf import TFPAEModel


def WHuber(y_true, y_pred, *, model: "TFPAEModel"):
    error = model._loss.input_mask * (y_true - y_pred) / model._loss.input_d_amp

    cond = tf.keras.backend.abs(error) < model.loss_clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = model.loss_clip_delta * (
        tf.keras.backend.abs(error) - 0.5 * model.loss_clip_delta
    )

    return tf.reduce_mean(
        tf.reduce_sum(tf.where(cond, squared_loss, linear_loss), axis=(-2, -1))
    )
