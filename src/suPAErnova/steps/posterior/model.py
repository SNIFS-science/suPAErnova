# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, get_args, override
from pathlib import Path  # noqa: TC003

from pydantic import BaseModel, PositiveInt, PositiveFloat  # noqa: TC002

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.data import SNPAEData  # noqa: TC001

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.nflow import (
        Model as NFLOW,
        ModelConfig as NFLOWConfig,
    )
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.nflow.model import NFlowModel

    from .posterior import Model, ModelConfig


class Stage(BaseModel):
    stage: PositiveInt
    name: str
    fname: str
    savepath: "Path | None" = None
    loadpath: "Path | None" = None

    debug: bool

    n_chains: int

    train_data: SNPAEData
    test_data: SNPAEData
    val_data: SNPAEData


class PosteriorModel[M: "Model", C: "ModelConfig"](SNPAEStep[C]):
    id: ClassVar[str] = "posterior_model"

    def __init__(self, config: C) -> None:
        self.model: M

        # --- Superclass Variables ---
        self.options: C
        self.config: "GlobalConfig"
        self.paths: "PathConfig"
        self.log: "Logger"
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool
        self.savepath: "Path"

        # --- Previous Step Variables ---
        self.nflow: "NFlowModel[NFLOW, NFLOWConfig]"
        self.train_data: SNPAEData
        self.test_data: SNPAEData
        self.val_data: SNPAEData

        # --- Setup Variables ---
        self.n_chains_early: int = self.options.n_chains_early
        self.n_chains_mid: int = self.options.n_chains_mid
        self.n_chains_final: int = self.options.n_chains_final

        self.stage_init: Stage
        self.stage_early: Stage
        self.stage_mid: Stage
        self.stage_final: Stage
        self.run_stages: list[Stage]

    @override
    def _setup(
        self,
        *,
        nflow: "NFlowModel[NFLOW, NFLOWConfig]",
    ) -> None:
        self.debug = self.options.debug

        self.nflow = nflow
        self.nflow.load()

        self.train_data = self.nflow.pae.train_data
        self.test_data = self.nflow.pae.test_data
        self.val_data = self.nflow.pae.val_data

        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)
        self.savepath = self.paths.out / self.model.name

        # --- Stages ---
        stage_data = {
            "train_data": self.train_data,
            "test_data": self.test_data,
            "val_data": self.val_data,
            "debug": self.debug,
        }
        self.stage_init = Stage.model_validate({
            "stage": 1,
            "name": "init",
            "fname": "init",
            "n_chains": 1,
            **stage_data,
        })
        self.stage_early = Stage.model_validate({
            "stage": 2,
            "name": "early",
            "fname": "early",
            "n_chains": self.n_chains_early,
            **stage_data,
        })
        self.stage_mid = Stage.model_validate({
            "stage": 3,
            "name": "mid",
            "fname": "mid",
            "n_chains": self.n_chains_mid,
            **stage_data,
        })
        self.stage_final = Stage.model_validate({
            "stage": 4,
            "name": "final",
            "fname": "final",
            "n_chains": self.n_chains_final,
            **stage_data,
        })
        self.run_stages = [
            self.stage_init,
            self.stage_early,
            self.stage_mid,
            self.stage_final,
        ]

    @override
    def _completed(self) -> bool:
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)
        savepath = self.savepath / self.model.model_path

        if not savepath.exists():
            self.log.debug(
                f"{self.name} has not completed as {savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)

        self.log.debug(f"Loading final Posterior model weights from {self.savepath}")
        self.model.load_checkpoint(self.savepath)

    @override
    def _run(self) -> None:
        model_cls = get_args(self.__orig_class__)[0]
        savepath: Path | None = None
        self.model = model_cls(self)
        for i, stage in enumerate(self.run_stages):
            self.log.debug(f"Starting Stage {i}: {stage.name}")
            if savepath is not None:
                stage.loadpath = savepath
            savepath = self.paths.out / self.model.name / stage.fname
            stage.savepath = savepath

            model_path = savepath / self.model.model_path
            weights_path = savepath / self.model.weights_path
            if model_path.exists() and not self.force:
                # Don't retrain stages if you don't need to
                self.log.debug(f"Loading stage weights from {weights_path}")
                self.model.stage = stage
                self.model.load_checkpoint(savepath)
            else:
                self.model.train_model(stage)
            self.model.save_checkpoint(savepath)
        print(self.model.chain_min)
        print(self.model.converged)
        print(self.model.num_evals)
        print(self.model.neg_log_like)
        print(self.model.delta_m_val)
        print(self.model.init_delta_m_val)
        print(self.model.delta_p_val)
        print(self.model.init_delta_p_val)
        print(self.model.us_val)
        print(self.model.init_us_val)
        print(self.model.bias_val)
        print(self.model.init_bias_val)

    @override
    def _result(self) -> None:
        model_cls = get_args(self.__orig_class__)[0]
        self.model = model_cls(self)

        self.log.debug(f"Saving final Posterior model weights to {self.savepath}")
        self.model.save_checkpoint(self.savepath)

    @override
    def _analyse(self) -> None:
        pass

    #
    # === Posterior Specific Functions ===
    #
