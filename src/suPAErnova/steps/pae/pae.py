# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from suPAErnova.steps.model import AbstractModelStep
from suPAErnova.configs.steps.data import DataStepResult

from .model import PAEModelStep

if TYPE_CHECKING:
    from logging import Logger

    from pydantic import PositiveFloat

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.configs.steps.pae import PAEStepConfig


class PAEStep[Backend: str](AbstractModelStep[Backend, PAEModelStep[Backend]]):
    # --- Class Variables ---
    id: ClassVar[str] = "pae"

    def __init__(self, config: "PAEStepConfig[Backend]") -> None:
        # --- Superclass Variables ---
        self.options: PAEStepConfig[Backend]
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
        self.train_data: list[DataStepResult]
        self.test_data: list[DataStepResult]
        self.val_data: list[DataStepResult]
        self.n_models: int
        self.n_kfolds: int

        self.seed: int = self.options.seed
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

    @override
    def _setup(self, *, data: "DataStep") -> None:
        super()._setup()

        # --- Previous Step Variables ---
        self.data = data
        self.validation_frac = self.options.validation_frac

        # --- Models ---
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
                DataStepResult.model_validate({
                    k: v[-ind_split:] for k, v in d.model_dump().items()
                })
                for d in train_data
            ]
            train_data = [
                DataStepResult.model_validate({
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

        for i, model in enumerate(self.models):
            model.setup(
                data=self.data,
                train_data=self.train_data[i],
                test_data=self.test_data[i],
                val_data=self.val_data[i],
            )


PAEStep.register_step()
