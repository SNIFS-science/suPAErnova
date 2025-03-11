import keras as ks

scheduler = {
    "": lambda lr, _: lr,
    "EXPONENTIAL": lambda lr, kwargs: ks.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=kwargs["lr_decay_steps"],
        decay_rate=kwargs["lr_decay_rate"],
    ),
}
