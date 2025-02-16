from typing import TYPE_CHECKING, Any

from keras import layers
import tensorflow as tf

from suPAErnova.model_utils.tf_layers import reduce_max, reduce_min, reduce_sum

if TYPE_CHECKING:
    import keras as ks
    import numpy as np
    from numpy.typing import NDArray

    from suPAErnova.models.tf_autoencoder import TFAutoencoder


# TODO: Choices
# choice=[
#     "MAE",
#     "WMAE",
#     "MSE",
#     "WMSE",
#     "RMSE",
#     "WRMSE",
#     "NGLL",
#     "HUBER",
#     "WHUBER",
#     "MAGNITUDE",
# ],


def MAE(
    x: "tf.Tensor",
    x_pred: "tf.Tensor",
    kwargs: dict[str, tf.Tensor],
):
    return tf.reduce_sum(
        tf.reduce_sum(tf.abs(kwargs["mask"] * (x - x_pred)), axis=(-2, -1)),
    )


loss_functions = {
    "MAE": MAE,
}


def apply_gradients(
    optimiser: "ks.Optimizer",
):
    # Because the optimiser creates variables internally
    # We run into issues with the tf.function optimiser
    # Since variables should only be created once
    # And then stored between runs
    # So we just make a new function for each optimiser
    @tf.function
    def _apply_gradients(
        model: "TFAutoencoder",
        x: "tf.Tensor",
        cond: "tf.Tensor",
        sigma: "tf.Tensor",
        mask: "tf.Tensor",
    ):
        with tf.GradientTape() as tape:
            loss, loss_terms = compute_loss(model, x, cond, sigma, mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(
            zip(gradients, model.trainable_variables, strict=True),
        )
        return loss, loss_terms

    return _apply_gradients


@tf.function
def compute_loss(
    model: "TFAutoencoder",
    x: "tf.Tensor",
    cond: "tf.Tensor",
    sigma: "tf.Tensor",
    mask: "tf.Tensor",
):
    loss_terms = []

    # Encode into latent parameters
    z = model.encode(x, cond, mask)
    # Decode from latent parameters
    x_pred = model.decode(z, cond, mask)

    loss = loss_functions[model.loss_fn](
        x,
        x_pred,
        {"sigma": sigma, "mask": mask, "model": model},
    )
    loss_terms.append(loss)

    # TODO: loss_recon
    # TODO: loss_cov

    return loss, loss_terms
