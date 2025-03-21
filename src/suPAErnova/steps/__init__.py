# Copyright 2025 Patrick Armstrong

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast
from pathlib import Path
import pkgutil
import importlib

if TYPE_CHECKING:
    from typing import TypeVar
    from logging import Logger
    from collections.abc import Callable

    from suPAErnova.configs import SNPAEConfig
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.config import GlobalConfig

    Step = TypeVar("Step", bound="SNPAEStep[Any]")
    StepReturn = TypeVar("StepReturn")


def callback(fn: "Callable[[Step], StepReturn]"):
    def wrapper(self: "Step") -> "StepReturn":
        callbacks = cast(
            "dict[str, Callable[[Step], None]]",
            self.options.callbacks.get(fn.__name__.lower(), {}),
        )
        pre_callback = callbacks.get("pre")
        if pre_callback is not None:
            pre_callback(self)
        rtn = fn(self)
        post_callback = callbacks.get("post")
        if post_callback is not None:
            post_callback(self)
        return rtn

    return wrapper


class SNPAEStep[Config: "SNPAEConfig"]:
    # Class Variables
    steps: ClassVar["dict[str, type[SNPAEStep[Any]]]"] = {}
    name: ClassVar["str"] = "step"

    @classmethod
    def register_step(cls) -> None:
        cls.steps[cls.name] = cls

    @staticmethod
    def register_steps() -> None:
        for _, module, is_pkg in pkgutil.iter_modules([str(Path(__file__).parent)]):
            if is_pkg:
                importlib.import_module(f"{__name__}.{module}")

    def __init__(self, config: "Config") -> None:
        self.options: Config = config
        self.config: GlobalConfig = config.config
        self.paths: PathConfig = config.paths
        self.log: Logger = config.log
        self.force: bool = self.config.force
        self.verbose: bool = self.config.verbose

    @abstractmethod
    def _setup(self) -> None:
        pass

    @callback
    def setup(self) -> None:
        self.log.info(f"Setting up {self.__class__.__name__}")
        self._setup()
        self.log.info(f"Finished setting up {self.__class__.__name__}")

    @abstractmethod
    def _completed(self) -> bool:
        pass

    @callback
    def completed(self) -> bool:
        self.log.info(f"Checking if {self.__class__.__name__} has completed")
        completed = self._completed()
        self.log.info(
            f"{self.__class__.__name__} has {'' if completed else 'not '}completed"
        )
        return completed

    @abstractmethod
    def _load(self) -> None:
        pass

    @callback
    def load(self) -> None:
        self.log.info(f"Loading {self.__class__.__name__}")
        self._load()
        self.log.info(f"Finished loading {self.__class__.__name__}")

    @abstractmethod
    def _run(self) -> None:
        pass

    @callback
    def run(self) -> None:
        if self.force or not self.completed():
            self.log.info(f"Running {self.__class__.__name__}")
            self._run()
            self.log.info(f"Finshed running {self.__class__.__name__}")
        else:
            self.load()

    @abstractmethod
    def _result(self) -> None:
        pass

    @callback
    def result(self) -> None:
        if self.force or not self.completed():
            self.log.info(f"Saving {self.__class__.__name__} results")
            self._result()
            self.log.info(f"Finished saving {self.__class__.__name__} results")

    @abstractmethod
    def _analyse(self) -> None:
        pass

    @callback
    def analyse(self) -> None:
        self.log.info(f"Analysing {self.__class__.__name__}")
        self._analyse()
        self.log.info(f"Finished analysing {self.__class__.__name__}")
