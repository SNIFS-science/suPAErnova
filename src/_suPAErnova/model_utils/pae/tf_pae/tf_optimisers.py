from typing import TYPE_CHECKING

from tensorflow import keras as ks

if TYPE_CHECKING:
    from collections.abc import Callable

    from suPAErnova.utils.suPAErnova_types import CFG, CONFIG


def null(
    _lr: ks.optimizers.schedules.LearningRateSchedule | float,
    _kwargs: "CFG",
) -> None:
    return None


def adam(
    lr: ks.optimizers.schedules.LearningRateSchedule | float,
    _kwargs: "CFG",
) -> ks.optimizers.Optimizer:
    return ks.optimizers.Adam(learning_rate=lr)


def adamw(
    lr: ks.optimizers.schedules.LearningRateSchedule | float,
    kwargs: "CFG",
) -> ks.optimizers.Optimizer:
    wd_schedule = ks.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=kwargs["weight_decay_rate"],
        decay_steps=kwargs["lr_decay_steps"],
        decay_rate=kwargs["lr_decay_rate"],
    )
    optimiser = ks.optimizers.AdamW(learning_rate=lr, weight_decay=0.0)
    optimiser.weight_decay = wd_schedule(optimiser.iterations)

    return optimiser


def sgd(
    lr: ks.optimizers.schedules.LearningRateSchedule | float,
    _kwargs: "CFG",
) -> ks.optimizers.Optimizer:
    return ks.optimizers.SGD(learning_rate=lr, momentum=0.9)


optimiser: "CONFIG[Callable[[ks.optimizers.schedules.LearningRateSchedule | float, CFG], ks.optimizers.Optimizer]]" = {
    "NULL": null,
    "ADAM": adam,
    "ADAMW": adamw,
    "SGD": sgd,
}
