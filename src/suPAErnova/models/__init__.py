from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from suPAErnova.steps.data import Data
    from suPAErnova.steps.model import ModelStep
    from suPAErnova.utils.suPAErnova_types import CFG


class PAEModel:
    def __init__(self, step: "ModelStep", training_params: "CFG") -> None:
        self.data: Data = step.data
        self.params: CFG = step.params
        self.training_params: CFG = training_params


from suPAErnova.models.tf_autoencoder import TFAutoencoder as TF_AUTOENCODER

models: dict[str, type[PAEModel]] = {"TF_AUTOENCODER": TF_AUTOENCODER}
