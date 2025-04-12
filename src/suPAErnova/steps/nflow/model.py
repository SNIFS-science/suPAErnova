# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, final, get_args, override

from suPAErnova.steps import SNPAEStep

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.pae import (
        Model as PAE,
        ModelConfig as PAEConfig,
    )
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.pae.model import PAEModel
    from suPAErnova.configs.steps.nflow import ModelConfig

    from .nflow import Model


class NFlowModel[M: "Model", C: "ModelConfig"](SNPAEStep[C]):
    id: ClassVar[str] = "nflow_model"

    def __init__(self, config: C) -> None:
        self.model: M

        # --- Superclass Variables ---
        self.options: C
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool
        self.savepath: Path

        # --- Previous Step Variables ---
        self.pae: PAEModel[PAE, PAEConfig]

    def prep_model(self, *, force: bool = False) -> M:
        if not force:
            try:
                return self.model
            except AttributeError:
                pass
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)
        return self.model

    @override
    def _setup(
        self,
        *,
        pae: "PAEModel[PAE, PAEConfig]",
    ) -> None:
        self.debug = self.options.debug

        self.pae = pae
        self.pae.load()

        self.prep_model()
        self.savepath = self.paths.out / self.model.name

    @override
    def _completed(self) -> bool:
        self.prep_model()
        savepath = self.savepath / self.model.model_path

        if not savepath.exists():
            self.log.debug(
                f"{self.name} has not completed as {savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self.prep_model()

        self.log.debug(f"Loading final NFlow model weights from {self.savepath}")
        self.model.load_checkpoint(self.savepath)

    @override
    def _run(self) -> None:
        self.prep_model()
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
        self.prep_model()
        self.log.debug(f"Saving final NFlow model weights to {self.savepath}")
        self.model.save_checkpoint(self.savepath)

    @override
    def _analyse(self) -> None:
        pass

    #
    # === NFlowModel Specific Functions ===
    #
