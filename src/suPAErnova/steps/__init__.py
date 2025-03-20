from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar, ClassVar, cast

if TYPE_CHECKING:
    from logging import Logger
    from collections.abc import Callable

    from suPAErnova.configs import SNPAEConfig

    Step = TypeVar("Step", bound="SNPAEStep[Any]")
    Rtn = TypeVar("Rtn")


def callback(fn: "Callable[[Step], Rtn]"):
    def wrapper(self: "Step") -> "Rtn":
        callbacks = cast(
            "dict[str, Callable[[Step], None]]",
            self.config.callbacks.get(fn.__name__.lower(), {}),
        )
        pre_callback = callbacks.get("pre")
        if pre_callback is not None:
            pre_callback(self)
        rtn = fn(self)
        post_callback = callbacks.get("pre")
        if post_callback is not None:
            post_callback(self)
        return rtn

    return wrapper


class SNPAEStep[Config: SNPAEConfig]:
    # Class Variables
    steps: ClassVar["dict[str, type[SNPAEStep[SNPAEConfig]]]"] = {}
    name: ClassVar["str"] = "step"

    @classmethod
    def register_step(cls) -> None:
        cls.steps[cls.name] = cls

    def __init__(self, config: "Config") -> None:
        self.config: Config = config
        self.log: Logger = config.log

    @abstractmethod
    def _setup(self) -> None:
        return None

    @callback
    def setup(self) -> None:
        self.log.info(f"Setting up {self.__class__.__name__}")
        self._run()

    @abstractmethod
    def _run(self) -> None:
        return None

    @callback
    def run(self) -> None:
        self.log.info(f"Running {self.__class__.__name__}")
        self._run()

    @abstractmethod
    def _result(self) -> None:
        return None

    @callback
    def result(self) -> None:
        self.log.info(f"Storing {self.__class__.__name__} results")
        self._result()

    @abstractmethod
    def _analyse(self) -> None:
        return None

    @callback
    def analyse(self) -> None:
        self.log.info(f"Analysing {self.__class__.__name__}")
        self._analyse()
