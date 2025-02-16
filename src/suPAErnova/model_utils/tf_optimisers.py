from typing import Any

import keras as ks


def adam(lr: ks.optimizers.schedules.LearningRateSchedule | float, _: dict[str, Any]):
    return ks.optimizers.Adam(learning_rate=lr)


def adamw(
    lr: ks.optimizers.schedules.LearningRateSchedule | float,
    kwargs: dict[str, Any],
):
    wd_schedule = ks.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=kwargs["weight_decay_rate"],
        decay_steps=kwargs["lr_decay_steps"],
        decay_rate=kwargs["lr_decay_rate"],
    )
    optimiser = ks.optimizers.AdamW(learning_rate=lr, weight_decay=0.0)
    optimiser.weight_decay = wd_schedule(optimiser.iterations)

    return optimiser


def sgd(lr: ks.optimizers.schedules.LearningRateSchedule | float, _: dict[str, Any]):
    return ks.optimizers.SGD(learning_rate=lr, momentum=0.9)


optimiser = {"ADAM": adam, "ADAMW": adamw, "SGD": sgd}
