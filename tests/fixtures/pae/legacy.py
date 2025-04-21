import os
from typing import TYPE_CHECKING
import argparse

import numpy as np
import pytest
import tensorflow as tf
from supaernova_legacy.utils import data_loader
from supaernova_legacy.models import (
    loader as model_loader,
    losses,
    autoencoder,
    autoencoder_training,
)
from supaernova_legacy.utils.YParams import YParams

from suPAErnova.configs.steps.pae import PAEStepResult

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable


def legacy_pae_step(
    data_params: dict[str, "Any"], pae_params: dict[str, "Any"]
) -> dict[str, "Any"]:
    # Except where indicated, this is taken verbatim from the `train_ae` script
    # Comments have been removed
    # Print statements have been removed
    # Plotting has been removed
    # Unused variables have been removed

    # Set model Architecture and training params and train
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="../config/train.yaml", type=str)
    parser.add_argument("--config", default="pae", type=str)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)
    epochs_initial = params["epochs"]

    train_data = data_loader.load_data(
        os.path.join(
            params["PROJECT_DIR"],
            params["train_data_file"],
        ),
        set_data_min_val=params["set_data_min_val"],
    )

    test_data = data_loader.load_data(
        os.path.join(
            params["PROJECT_DIR"],
            params["test_data_file"],
        ),
        set_data_min_val=params["set_data_min_val"],
    )

    # Mask certain supernovae
    train_data["mask_sn"] = data_loader.get_train_mask(train_data, params)
    test_data["mask_sn"] = data_loader.get_train_mask(test_data, params)

    # Mask certain spectra
    train_data["mask_spectra"] = data_loader.get_train_mask_spectra(train_data, params)
    test_data["mask_spectra"] = data_loader.get_train_mask_spectra(test_data, params)

    # Split off validation set from training set
    if params["use_val"]:
        train_data, val_data = data_loader.split_train_and_val(train_data, params)
    else:
        val_data = test_data

    for il, latent_dim in enumerate(params["latent_dims"]):
        params["latent_dim"] = latent_dim
        params["num_training_stages"] = latent_dim + 3
        params["train_stage"] = 0

        if latent_dim == 0:
            # Model parameters are (\Delta t, \Delta m, \Delta A_v)
            # train \Delta m and \Delta A_v first. Then \Delta t
            params["train_stage"] = 1
            params["num_training_stages"] = 2

        tf.random.set_seed(params["seed"])

        # Create model
        AEmodel = autoencoder.AutoEncoder(params, training=True)

        # Model Summary
        if params["model_summary"] and (il == 0):
            print("Encoder Summary")
            AEmodel.encoder.summary()

            print("Decoder Summary")
            AEmodel.decoder.summary()

        print(f"Training model with {latent_dim:d} latent dimensions")
        print("Running training stage ", params["train_stage"])
        # Train model, splitting into seperate training stages for seperate model parameters, if desired.
        _training_loss, _val_loss, _test_loss = autoencoder_training.train_model(
            train_data,
            val_data,
            test_data,
            AEmodel,
        )

        params["train_stage"] += 1
        if not params["train_latent_individual"]:
            params["train_stage"] += params["latent_dim"] - 1

        while params["train_stage"] < params["num_training_stages"]:
            print("Running training stage ", params["train_stage"])

            epochs_initial = params["epochs"]

            AEmodel_second = autoencoder.AutoEncoder(params, training=True)
            if params["train_stage"] < params["num_training_stages"] - 2:
                AEmodel_second.params["epochs"] = params["epochs_latent"]
            if (
                params["train_stage"] >= params["num_training_stages"] - 2
            ):  # add in delta mag
                AEmodel_second.params["epochs"] = params["epochs_final"]

            # Load best checkpoint from step 0 training
            encoder, decoder, _AE_params = model_loader.load_ae_models(params)

            final_dense_layer = len(params["encode_dims"]) + 4

            final_layer_weights = encoder.layers[final_dense_layer].get_weights()[0]
            final_layer_weights_init = AEmodel_second.encoder.layers[
                final_dense_layer
            ].get_weights()[0]

            if params["train_stage"] <= params["latent_dim"]:  # add in z_1, ..., z_n
                idim = 2 + params["train_stage"]
                final_layer_weights[:, idim] = final_layer_weights_init[:, idim] / 100

            if (
                params["train_stage"] == params["num_training_stages"] - 2
            ):  # add in delta mag
                final_layer_weights[:, 1] = final_layer_weights_init[:, 1] / 100
                if not params["train_latent_individual"]:
                    final_layer_weights[:, 3:] = final_layer_weights_init[:, 3:] / 100

            if (
                params["train_stage"] == params["num_training_stages"] - 1
            ):  # add in delta t
                final_layer_weights[:, 0] = final_layer_weights_init[:, 0] / 100

            encoder.layers[final_dense_layer].set_weights([final_layer_weights])

            AEmodel_second.encoder.set_weights(encoder.get_weights())
            AEmodel_second.decoder.set_weights(decoder.get_weights())

            _training_loss, _val_loss, _test_loss = autoencoder_training.train_model(
                train_data,
                val_data,
                test_data,
                AEmodel_second,
            )

            params["train_stage"] += 1
    return {}


@pytest.fixture(scope="session")
def legacy_pae_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]":
    def _legacy_pae_step(
        data_params: dict[str, "Any"], pae_params: "dict[str, Any]"
    ) -> "dict[str, Any]":
        data_params["data_path"] = data_path
        data_params["root_path"] = root_path
        data_params["cache_path"] = cache_path

        pae_params["data_path"] = data_path
        pae_params["root_path"] = root_path
        pae_params["cache_path"] = cache_path
        return legacy_pae_step(data_params, pae_params)

    return _legacy_pae_step


@pytest.fixture(scope="session")
def legacy_pae_result_factory(
    legacy_pae_step_factory: "Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]",
) -> "Callable[[dict[str, Any], dict[str, Any]], PAEStepResult]":
    def _legacy_pae_result(
        data_params: dict[str, "Any"], pae_params: dict[str, "Any"]
    ) -> "PAEStepResult":
        return PAEStepResult.model_validate(
            legacy_pae_step_factory(data_params, pae_params)
        )

    return _legacy_pae_result
