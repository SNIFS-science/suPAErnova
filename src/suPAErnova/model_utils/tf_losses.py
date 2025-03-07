from typing import TYPE_CHECKING

import tensorflow as tf
import tensorflow_probability as tfp

if TYPE_CHECKING:
    from tensorflow import keras as ks

    from suPAErnova.models.tf_autoencoder import TFAutoencoder

# TODO: Choices
# choice=[
#     "MSE",
#     "WMSE",
#     "RMSE",
#     "WRMSE",
#     "NGLL",
#     "HUBER",
#     "WHUBER",
#     "MAGNITUDE",
# ],


def MAE(x: "tf.Tensor", x_pred: "tf.Tensor", kwargs: dict[str, tf.Tensor]):
    return tf.reduce_sum(
        tf.reduce_sum(
            kwargs["mask"] * tf.abs(x - x_pred),
            axis=(-2, -1),
        ),
    )


def WMAE(x: "tf.Tensor", x_pred: "tf.Tensor", kwargs: dict[str, tf.Tensor]):
    return tf.reduce_sum(
        tf.reduce_sum(
            kwargs["mask"] * tf.abs((x - x_pred) / kwargs["sigma"]),
            axis=(-2, -1),
        ),
    )


loss_functions = {
    "MAE": MAE,
    "WMAE": WMAE,
}


def apply_gradients(
    optimiser: "ks.Optimizer",
    model: "TFAutoencoder",
):
    # Because the optimiser creates variables internally
    # We run into issues with the tf.function optimiser
    # Since variables should only be created once
    # And then stored between runs
    # So we just make a new function for each optimiser
    @tf.function
    def _apply_gradients(
        flux: "tf.Tensor",
        time: "tf.Tensor",
        sigma: "tf.Tensor",
        mask: "tf.Tensor",
    ):
        with tf.GradientTape() as tape:
            loss, loss_terms = compute_loss(model, flux, time, sigma, mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(
            zip(gradients, model.trainable_variables, strict=True),
        )
        return loss_terms

    return _apply_gradients


@tf.function
def compute_loss(
    model: "TFAutoencoder",
    flux: "tf.Tensor",
    time: "tf.Tensor",
    sigma: "tf.Tensor",
    mask: "tf.Tensor",
):
    # Encode into latent parameters
    z = model.encode(flux, time, mask)
    # Decode from latent parameters
    flux_pred = model.decode(z, time, mask)

    # Since the mask needs to be multiplied with many other tensors
    # Cast it to float32 now
    mask = tf.cast(mask, tf.float32)

    loss = loss_functions[model.loss_fn](
        flux,
        flux_pred,
        {"sigma": sigma, "mask": mask, "model": model},
    )

    # Overall loss, reconstruction loss, and (optionally) covariance loss
    loss_terms = [None, loss, None]

    if model.loss_amplitude_offset > 0:
        loss_amplitude_offset = tf.reduce_mean(
            tf.abs(tf.reduce_sum((flux - flux_pred) * mask, axis=(-2, -1))),
        )
        loss += loss_amplitude_offset * model.loss_amplitude_offset

    if model.loss_amplitude_parameter > 0:
        z_median = tfp.stats.percentile(z[:, 0], 50, interpolation="midpoint")
        loss_amplitude = (1 - z_median) ** 2
        loss += loss_amplitude * model.loss_amplitude_parameter

    if model.loss_covariance > 0:
        z_cov = z
        is_kept = tf.reduce_min(mask, axis=-1, keepdims=True)
        is_kept = tf.reduce_max(is_kept, axis=-2)

        num_kept = tf.reduce_sum(is_kept)
        mean_z = tf.reduce_sum(z_cov * is_kept, axis=0, keepdims=True) / num_kept

        z_cov = (z_cov - mean_z) * is_kept

        cov_z = tf.matmul(tf.transpose(z_cov), z_cov) / num_kept
        std_z = tf.sqrt(tf.reduce_sum(z_cov**2, axis=0) / num_kept)
        std_z = tf.where(std_z < 1e-3, tf.ones(std_z.shape[0]), std_z)

        cov_z /= std_z

        istart = 2
        iend = cov_z.shape[0]
        cov_mask = 1 - tf.eye(cov_z.shape[0])
        if model.decorrelate_dust:
            istart += 1
        if not model.decorrelate_all:
            cov_mask[istart:iend, istart:iend] = (
                0.0  # Remove correlation from central region
            )

        loss_cov = tf.reduce_sum(
            tf.square(tf.math.multiply(cov_z, cov_mask)),
        ) / tf.reduce_sum(cov_mask)

        loss += loss_cov * model.loss_covariance

        loss_terms[-1] = loss_cov * model.loss_covariance

    if model.kernel_regulariser:
        loss += tf.reduce_sum(model.losses)

    loss_terms[0] = loss

    return loss, tf.stack(
        loss_terms,
        axis=0,
    )
