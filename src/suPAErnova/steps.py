from abc import abstractmethod
from logging import Logger
from pathlib import Path
import typing
from typing import Any, Callable, Generic, Literal, TypeVar, cast, final
from suPAErnova.utils.typing import CFG
import toml

# --- Types ---
type RequirementReturn[T] = tuple[Literal[False], str] | tuple[Literal[True], T]
IN = TypeVar("IN")
OUT = TypeVar("OUT")


@final
class Requirement(Generic[IN, OUT]):
    def __init__(
        self,
        name: str,
        description: str,
        choice: list[IN] | None = None,
        bounds: tuple[IN, IN] | None = None,
        transform: Callable[[IN, CFG, CFG], OUT] | None = None,
        valid_transform: Callable[[OUT, CFG, CFG], RequirementReturn[OUT]]
        | None = None,
    ):
        self.name = name
        self.description = description
        self.type = type
        self.choice = choice
        self.bounds = bounds
        self.transform = transform
        self.valid_transform = valid_transform

    def _IN(self):
        return typing.get_args(self.__orig_class__)[0]

    def _OUT(self):
        return typing.get_args(self.__orig_class__)[-1]

    def validate_type(self, opt: object) -> RequirementReturn[IN]:
        if not isinstance(opt, self._IN()):
            return (
                False,
                f"Incorrect type {type(opt)}, must be {self._IN()}",
            )
        # If isinstance(opt, IN), cast to IN
        return True, cast(IN, opt)

    def validate_choice(self, opt: IN) -> RequirementReturn[IN]:
        if self.choice is not None and opt not in self.choice:
            return (
                False,
                f"Unknown choice {opt}, must be one of {self.choice}",
            )
        return True, opt

    def validate_bounds(self, opt: IN) -> RequirementReturn[IN]:
        if self.bounds is not None and not (self.bounds[0] <= opt <= self.bounds[-1]):
            return (
                False,
                f"{opt} must be within {self.bounds}",
            )
        return True, opt

    def validate_transform(
        self, opt: IN, cfg: CFG, opts: CFG
    ) -> RequirementReturn[IN | OUT]:
        if self.transform is not None:
            try:
                result = self.transform(opt, cfg, opts)
                return True, result
            except Exception as e:
                return False, f"Error tranforming {opt}: {e}"
        return True, opt

    def validate_post_transform(
        self, opt: IN | OUT, cfg: CFG, opts: CFG
    ) -> RequirementReturn[IN | OUT]:
        if self.transform is not None and self.valid_transform is not None:
            try:
                # opt is of type OUT if transform != None
                return self.valid_transform(cast(OUT, opt), cfg, opts)
            except Exception as e:
                return False, f"Error validating transform {opt}: {e}"
        return True, opt

    def validate(self, opt: IN, cfg: CFG, opts: CFG):
        ok, result = self.validate_type(opt)
        if not ok:
            return ok, result
        ok, result = self.validate_choice(opt)
        if not ok:
            return ok, result
        ok, result = self.validate_bounds(opt)
        if not ok:
            return ok, result
        ok, result = self.validate_transform(opt, cfg, opts)
        if not ok:
            return ok, result
        # result is of type IN | OUT if transform did not fail
        ok, result = self.validate_post_transform(cast(IN | OUT, result), cfg, opts)
        if not ok:
            return ok, result
        return (ok, result)


type REQ = Requirement[Any, Any]


class Step:
    required: list[REQ] = []
    optional: list[REQ] = []

    def __init__(self, config: CFG):
        self.name: str = self.__class__.__name__.upper()

        self.config: CFG = config
        self.cfg: CFG = config["global"]
        self.opts: CFG = config[self.name]

        self.outpath: Path = self.opts.get("output", self.cfg["output"] / self.name)
        if not self.outpath.is_absolute():
            self.outpath = self.cfg["output"] / self.outpath
        if not self.outpath.exists():
            self.outpath.mkdir(parents=True)
        self.results: CFG = {"output": self.outpath}

        self.log: Logger = self.cfg["log"]
        self.log.debug(f"Running {self.name} with opts: {self.opts}")

        self.results["config"] = self.outpath / f"{self.name}.toml"
        self.log.debug(f"Writing {self.name} opts to {self.results['config']}")
        with open(self.results["config"], "w") as io:
            _ = toml.dump({self.name: self.opts}, io)

        is_valid = self.validate()
        if not is_valid:
            raise ValueError(f"Invalid {self.name} configuration")

    def validate(self):
        for requirement in self.required:
            key = requirement.name
            opt = self.opts.get(key)
            if opt is None:
                self.log.error(
                    f"{self.name} is missing required option: {key}: {requirement.description}"
                )
                return False
            else:
                ok, result = requirement.validate(opt, self.cfg, self.opts)
            if not ok:
                self.log.error(f"Invalid `{key}`=`{opt}`: {result}")
                return False
            self.opts[key] = result

        for requirement in self.optional:
            key = requirement.name
            opt = self.opts.get(key)
            if opt is not None:
                ok, result = requirement.validate(opt, self.cfg, self.opts)
                if not ok:
                    self.log.error(f"Invalid `{key}`=`{opt}`: {result}")
                    return False
                self.opts[key] = result
        return True

    @abstractmethod
    def _run(self) -> RequirementReturn[None]:
        return True, None

    def run(self):
        self.log.info(f"Running {self.name}")
        try:
            ok, result = self._run()
        except Exception as e:
            ok = False
            result = e
        if not ok:
            self.log.error(f"Error running {self.name}: {result}")
        return self

    @abstractmethod
    def _result(self) -> RequirementReturn[None]:
        return True, None

    def result(self):
        try:
            ok, result = self._result()
        except Exception as e:
            ok = False
            result = e
        if not ok:
            self.log.error(f"Error getting results of {self.name}: {result}")
        self.results["opts"] = self.opts
        self.cfg["results"][self.name] = self.results
        self.config["global"] = self.cfg
        self.config[self.name] = self.opts
        self.log.info(f"Finished running {self.name}")
        return self.config
