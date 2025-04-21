# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, override
import importlib

from suPAErnova.steps.backends import AbstractModel

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.pae.model import PAEModel
    from suPAErnova.configs.steps.nflow.model import NFlowModelConfig

    from .tf import TFNFlowModel
    from .tch import TCHNFlowModel

    NFlowModel = TFNFlowModel | TCHNFlowModel


class NFlowModelStep[Backend: str](AbstractModel[Backend]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[NFlowModel]]"]] = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFNflowModel,
        "PyTorch": lambda: importlib.import_module(".tch", __package__).TCHNFlowModel,
    }
    id: ClassVar[str] = "nflow_model"

    def __init__(self, config: "NFlowModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: NFlowModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool
        self.savepath: Path

        self.pae: PAEModel

    @override
    def _setup(self, *, pae: "PAEModel") -> None:
        self.debug = self.options.debug

        self.pae = pae
        self.pae.load()

        self._model()
        self.savepath = self.paths.out / self.model.name

    @override
    def _completed(self) -> bool:
        self._model()
        savepath = self.savepath / self.model.model_path

        if not savepath.exists():
            self.log.debug(
                f"{self.name} has not completed as {savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self._model()

        self.log.debug(f"Loading final NFlow model weights from {self.savepath}")
        self.model.load_checkpoint(self.savepath)

    @override
    def _run(self) -> None:
        self._model()
        model_path = self.savepath / self.model.model_path
        weights_path = self.savepath / self.model.weights_path
        if model_path.exists() and not self.force:
            # Don't retrain stages if you don't need to
            self.log.debug(f"Loading weights from {weights_path}")
            self.model.load_checkpoint(self.savepath)
        else:
            self.model.train_model(savepath=self.savepath)
        self.model.save_checkpoint(self.savepath)

    @override
    def _result(self) -> None:
        self._model()
        self.log.debug(f"Saving final NFlow model weights to {self.savepath}")
        self.model.save_checkpoint(self.savepath)

    @override
    def _analyse(self) -> None:
        pass

    #
    # === NFlowModel Specific Functions ===
    #
