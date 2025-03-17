from typing import TYPE_CHECKING, Any

from tensorflow import keras as ks

if TYPE_CHECKING:
    from collections.abc import Callable

    from suPAErnova.utils.suPAErnova_types import CONFIG


def null(_lr: float, _kwargs: "CONFIG[Any]") -> None:
    return None


def exponential(
    lr: float,
    kwargs: "CONFIG[Any]",
) -> ks.optimizers.schedules.LearningRateSchedule:
    return ks.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=kwargs["lr_decay_steps"],
        decay_rate=kwargs["lr_decay_rate"],
    )


def identity(lr: float, _kwargs: "CONFIG[Any]") -> float:
    return lr


scheduler: "CONFIG[Callable[[float, dict[str, Any]], ks.optimizers.schedules.LearningRateSchedule | float]]" = {
    "NULL": null,  # pyright:ignore[reportAssignmentType]
    "IDENTITY": identity,
    "EXPONENTIAL": exponential,
}
