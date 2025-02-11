from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self, TypeVar
from pathlib import Path
import traceback
from collections.abc import Callable

import toml
from tqdm import tqdm

import suPAErnova.analysis as analyses
from suPAErnova.config.requirements import Requirement

if TYPE_CHECKING:
    from typing import ClassVar
    from logging import Logger
    from collections.abc import Sequence

    from suPAErnova.utils.typing import CFG
    from suPAErnova.config.requirements import REQ, RequirementReturn


# === Optional Requirements ===
# --- Force Run ---
force = Requirement[bool, bool](
    name="force",
    description="Force rerun step, overwriting previous run",
    default=False,
)

analysis = Requirement[dict, dict](
    name="analysis",
    description="Plotting and analysis options",
    default={},
)


def get_callbacks(callbacks: dict, cfg: "CFG", _2: "CFG"):
    rtn = {}
    for fn, script in callbacks.items():
        rtn[fn] = {}
        path = Path(script)
        if not path.is_absolute():
            path = cfg["BASE"] / path
        if not path.exists():
            return False, f"{path} does not exist"
        with path.open("r") as io:
            script_code = io.read()
        # Create an isolated namespace
        local_scope = {}
        exec(script_code, globals(), local_scope)
        if "pre" in local_scope:
            if not isinstance(local_scope["pre"], Callable):
                return False, f"pre-{script} is not callable"
            rtn[fn]["pre"] = local_scope["pre"]
        if "post" in local_scope:
            if not isinstance(local_scope["post"], Callable):
                return False, f"post-{script} is not callable"
            rtn[fn]["post"] = local_scope["post"]
    return True, rtn


callbacks = Requirement[dict, dict](
    name="callbacks",
    description="Path to scripts containing callback functions",
    default={},
    transform=get_callbacks,
)


class Callback:
    def __init__(self) -> None:
        self.callbacks: dict[str, dict[str, Callable[[Self], None]]] = {}


S = TypeVar("S", bound=Callback)


def callback(fn: "Callable[[S], Any]"):
    def wrapper(self: S):
        callbacks = self.callbacks.get(fn.__name__.upper(), {})
        pre_callback = callbacks.get("pre")
        if pre_callback is not None:
            pre_callback(self)
        rtn = fn(self)
        post_callback = callbacks.get("post")
        if post_callback is not None:
            post_callback(self)
        return rtn

    return wrapper


class Step(Callback):
    required: "ClassVar[list[REQ]]" = []
    optional: "ClassVar[list[REQ]]" = [force, analysis, callbacks]
    prev: "ClassVar[list[str]]" = []

    @classmethod
    def __update__(cls) -> None:
        if cls is Step:
            return
        parent_cls = next(
            (
                parent
                for parent in cls.__bases__
                if issubclass(parent, Step) and parent is not Step
            ),
            Step,
        )
        # Create new lists instead of mutating to avoid side effects
        cls.required = parent_cls.required + cls.required
        cls.optional = parent_cls.optional + cls.optional
        cls.prev = parent_cls.prev + cls.prev

    def __init_subclass__(cls, **kwargs: "CFG") -> None:
        super().__init_subclass__(**kwargs)
        cls.__update__()

    def __init__(self, config: "CFG") -> None:
        super().__init__()
        self.name: str = self.__class__.__name__.upper()

        self.orig_config: CFG = config
        self.global_cfg: CFG = config["GLOBAL"]
        self.opts: CFG = config[self.name]

        self.outpath: Path = self.opts.get(
            "OUTPUT",
            self.global_cfg["OUTPUT"] / self.name,
        )
        if not self.outpath.is_absolute():
            self.outpath = self.global_cfg["OUTPUT"] / self.outpath
        if not self.outpath.exists():
            self.outpath.mkdir(parents=True)

        self.log: Logger = self.global_cfg["LOG"]
        self.log.debug(f"Running {self.name} with opts: {self.opts}")

        self.configpath: Path = self.outpath / f"{self.name}.toml"
        self.plotpath: Path = self.outpath / "plots"
        if not self.plotpath.exists():
            self.plotpath.mkdir(parents=True)

        self.log.debug(f"Writing {self.name} opts to {self.configpath}")
        with self.configpath.open("w", encoding="utf-8") as io:
            _ = toml.dump({self.name: self.opts}, io)

        is_valid = self.validate()
        if not is_valid:
            msg = f"Invalid {self.name} configuration"
            raise ValueError(msg)

        self.force: bool = self.global_cfg["FORCE"] or self.opts["FORCE"]

        self.analyses: dict[str, Callable[[Self, CFG], None]] = getattr(
            analyses,
            self.name,
        ).ANALYSES

        self.callbacks: dict[str, dict[str, Callable[[Self], None]]] = self.opts[
            "CALLBACKS"
        ]
        # Bind the callback function to self
        # raw_callbacks: Callable[[Self], None] = self.opts["CALLBACKS"]
        # self.callbacks: Callable[[Self], None] = MethodType(raw_callbacks, self)

    @callback
    def validate(self) -> bool:
        for step in self.prev:
            if self.global_cfg["RESULTS"].get(step) is None:
                self.log.error(f"Attempting to run {self.name} before {step}")
                return False
        for requirement in self.required:
            key = requirement.name.upper()
            opt = self.opts.get(key)
            if opt is None:
                self.log.error(
                    f"{self.name} is missing required option: {key}: {requirement.description}",
                )
                return False
            ok, result = requirement.validate(opt, self.global_cfg, self.opts)
            if not ok:
                self.log.error(f"Invalid `{key}`=`{opt}`: {result}")
                return False
            self.opts[key] = result

        for requirement in self.optional:
            key = requirement.name.upper()
            opt = self.opts.get(key)
            if opt is None:
                opt = requirement.default
            if opt is not None:
                ok, result = requirement.validate(opt, self.global_cfg, self.opts)
                if not ok:
                    self.log.error(f"Invalid `{key}`=`{opt}`: {result}")
                    return False
            else:
                result = None
            self.opts[key] = result
        return True

    def tqdm(self, lst: "Sequence[Any]", *args, **kwargs):
        return lst if not self.global_cfg["VERBOSE"] else tqdm(lst, *args, **kwargs)

    @abstractmethod
    def _is_completed(self) -> bool:
        return False

    @abstractmethod
    def _load(self) -> None:
        pass

    @abstractmethod
    def _run(self) -> "RequirementReturn[None]":
        return True, None

    @callback
    def run(self):
        self.log.info(f"Running {self.name}")
        should_run = not self._is_completed()
        if self.force:
            self.log.debug(f"Forced running of {self.name}")
            should_run = True
        if should_run:
            try:
                ok, result = self._run()
            except Exception:
                ok = "Exception"
                result = traceback.format_exc()
                self.log.exception(f"Error running {self.name}: {result}")
            if not ok:
                self.log.error(f"Error running {self.name}: {result}")
                #       or save self.sne to file
        else:
            self.log.info(f"{self.name} already completed, loading previous result")
            self._load()
        return self

    @abstractmethod
    def _result(self) -> "RequirementReturn[None]":
        return True, None

    @callback
    def result(self):
        self.log.info(f"Storing {self.name} results")
        try:
            ok, result = self._result()
        except Exception:
            ok = "Exception"  # We handled the exception here so no need to log the error later
            result = traceback.format_exc()
            self.log.exception(f"Error getting results of {self.name}: {result}")
        if not ok:
            self.log.error(f"Error getting results of {self.name}: {result}")
        self.global_cfg["RESULTS"][self.name] = self
        self.orig_config["GLOBAL"] = self.global_cfg
        self.orig_config[self.name] = self.opts
        self.log.info(f"Finished running {self.name}")
        return self.orig_config

    @abstractmethod
    def _analyse(self) -> "RequirementReturn[None]":
        return True, None

    @callback
    def analyse(self) -> None:
        self.log.info(f"Analysing {self.name}")
        try:
            ok, result = self._analyse()
        except Exception:
            ok = "Exception"  # We handled the exception here so no need to log the error later
            result = traceback.format_exc()
            self.log.exception(f"Error analysing {self.name}: {result}")
        if not ok:
            self.log.error(f"Error analysing {self.name}: {result}")
        for key, opts in self.opts["ANALYSIS"].items():
            fn = self.analyses.get(key)
            if fn is None:
                self.log.error(f"Unknown analysis function: {key}")
            else:
                fn(self, opts)


from suPAErnova.steps.data import Data
from suPAErnova.steps.model import Model
from suPAErnova.steps.autoencoder import AutoEncoder

__all__ = ["AutoEncoder", "Data", "Model", "Step"]
