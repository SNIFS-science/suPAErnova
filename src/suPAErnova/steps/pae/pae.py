# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from suPAErnova.steps import SNPAEStep
from suPAErnova.steps.data import SNPAEData
from suPAErnova.steps.pae.tf import TFPAEModel
from suPAErnova.steps.pae.tch import TCHPAEModel
from suPAErnova.steps.pae.model import PAEModel
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.pae.tf.tf import TFPAEModelConfig
from suPAErnova.configs.steps.pae.tch.tch import TCHPAEModelConfig

if TYPE_CHECKING:
    from logging import Logger

    from pydantic import PositiveFloat

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig

    ModelConfig = TFPAEModelConfig | TCHPAEModelConfig
    Model = TFPAEModel[int, int, int, int, int] | TCHPAEModel

ModelMap: "dict[type[ModelConfig], type[Model]]" = {
    TFPAEModelConfig: TFPAEModel[int, int, int, int, int],
    TCHPAEModelConfig: TCHPAEModel,
}


class PAEStep(SNPAEStep[PAEStepConfig]):
    # Class Variables
    id: ClassVar["str"] = "pae"

    def __init__(self, config: "PAEStepConfig") -> None:
        # --- Superclass Variables ---
        self.options: PAEStepConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Previous Step Variables ---
        self.data: DataStep
        self.validation_frac: PositiveFloat

        # --- Setup Variables ---
        self.pae_models: list[PAEModel[Model]]
        self.train_data: list[SNPAEData]
        self.test_data: list[SNPAEData]
        self.val_data: list[SNPAEData]
        self.n_models: int
        self.n_kfolds: int

        self.seed: int = self.options.seed
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

    @override
    def _setup(self, *, data: "DataStep") -> None:
        # --- Previous Step Variables ---
        self.data = data
        self.validation_frac = self.options.validation_frac

        # --- Models ---
        self.pae_models = [
            PAEModel[ModelMap[model.__class__]](model) for model in self.options.models
        ]
        self.n_models = len(self.pae_models)
        self.n_kfolds = self.data.n_kfolds
        self.log.debug(
            f"Training {self.n_models} models across {self.n_kfolds} kfolds."
        )
        if self.n_models > self.n_kfolds:
            self.log.warning(
                f"Data has {self.n_kfolds} kfolds, but {self.n_models} models were requested, some models will share the same training, testing, and validation data."
            )

        # --- Data ---
        train_data = self.data.train_data
        test_data = self.data.test_data
        if self.validation_frac > 0:
            ind_split = int(self.data.sn_dim * self.validation_frac)
            val_data = [
                SNPAEData.model_validate({
                    k: v[-ind_split:] for k, v in d.model_dump().items()
                })
                for d in train_data
            ]
            train_data = [
                SNPAEData.model_validate({
                    k: v[:-ind_split] for k, v in d.model_dump().items()
                })
                for d in train_data
            ]
        else:
            val_data = test_data

        # `(list * ((desired_length // actual_length) + 1))[:desired_length]`
        # Repeat `list` `(desired_length // actual_length) + 1` times, then take the first `desired_length` items
        self.train_data = (train_data * ((self.n_models // self.n_kfolds) + 1))[
            : self.n_models
        ]
        self.test_data = (test_data * ((self.n_models // self.n_kfolds) + 1))[
            : self.n_models
        ]
        self.val_data = (val_data * ((self.n_models // self.n_kfolds) + 1))[
            : self.n_models
        ]

        for i, pae_model in enumerate(self.pae_models):
            pae_model.setup(
                data=self.data,
                train_data=self.train_data[i],
                test_data=self.test_data[i],
                val_data=self.val_data[i],
            )

    @override
    def _completed(self) -> bool:
        return all(pae_model.completed() for pae_model in self.pae_models)

    @override
    def _load(self) -> None:
        for _i, pae_model in enumerate(self.pae_models):
            pae_model.load()

    @override
    def _run(self) -> None:
        for _i, pae_model in enumerate(self.pae_models):
            pae_model.run()

    @override
    def _result(self) -> None:
        for _i, pae_model in enumerate(self.pae_models):
            pae_model.result()

    @override
    def _analyse(self) -> None:
        for _i, pae_model in enumerate(self.pae_models):
            pae_model.analyse()


PAEStep.register_step()
