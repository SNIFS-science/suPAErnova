# Copyright 2025 Patrick Armstrong
"""Generics for SuPAErnova PAE models."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from suPAErnova.steps.pae import PAEStep
    from suPAErnova.steps.data import DATAStep
    from suPAErnova.utils.suPAErnova_types import CFG


class PAEModel:
    """A generic Probabilitistic AutoEncoder model.

    Attributes:
        data (Data): The data used in training this model, produced by the Data step
        params (CFG): Parameters used when setting up this model
        training_params (CFG): Parameters used when training this model
    """

    def __init__(self, step: "PAEStep", training_params: "CFG") -> None:
        """Initialise a PAEModel.

        Args:
            step (ModelStep): The model step which is responsible for building and training this model
            training_params (CFG): Parameters used when training this model
        """
        self.data: DATAStep = step.data
        self.params: CFG = step.params
        self.training_params: CFG = training_params


from suPAErnova.models.pae.tf_pae import TF_PAEModel

models: dict[str, type["PAEModel"]] = {"TF_PAE": TF_PAEModel}
