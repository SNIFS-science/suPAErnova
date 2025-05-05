from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from suPAErnova.steps.pae.model import PAEModel
    from suPAErnova.configs.steps.pae import PAEStage
    from suPAErnova.configs.steps.pae.tch import TCHPAEModelConfig


class TCHPAEModel:
    def __init__(
        self,
        config: "PAEModel[TCHPAEModel, TCHPAEModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        self.name: str
        self.stage: PAEStage
        self.weights_path: str
        self.model_path: str

    def save_checkpoint(self, savepath: "Path") -> None:
        pass

    def train_model(
        self,
        _stage: "PAEStage",
    ) -> None:
        pass

    def load_checkpoint(
        self, loadpath: "Path", *, reset_weights: bool | None = None
    ) -> None:
        pass
