from typing import TYPE_CHECKING, Any

from tensorflow import keras as ks

if TYPE_CHECKING:
    from collections.abc import Callable

    from suPAErnova.utils.suPAErnova_types import CFG


def null(_lr: float, _kwargs: "CFG") -> None:
    return None


def exponential(
    lr: float,
    kwargs: "CFG",
) -> ks.optimizers.schedules.LearningRateSchedule:
    return ks.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=kwargs["lr_decay_steps"],
        decay_rate=kwargs["lr_decay_rate"],
    )


def identity(lr: float, _kwargs: "CFG") -> float:
    return lr


scheduler: "CONFIG[Callable[[float, CFG], Any]], ks.optimizers.schedules.LearningRateSchedule | float]]" = {
    "NULL": null,
    "IDENTITY": identity,
    "EXPONENTIAL": exponential,
}
