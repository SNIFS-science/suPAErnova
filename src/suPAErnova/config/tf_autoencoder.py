from typing import TYPE_CHECKING, Literal
from pathlib import Path
import itertools

from suPAErnova.config.requirements import Requirement

if TYPE_CHECKING:
    from suPAErnova.utils.typing import CFG
    from suPAErnova.config.requirements import REQ


# Data Settings
def valid_kfold(kfold: int | list[int] | Literal[True], _1: "CFG", _2: "CFG"):
    if kfold is not True:
        if isinstance(kfold, int):
            kfold = [kfold]
        if len(set(kfold)) != len(kfold):
            return (False, f"kfold: {kfold} contains duplicates")
    return (True, kfold)


kfold = Requirement[int | list[int] | bool, list[int] | bool](
    name="kfold",
    description="Which kfold(s) to use",
    default=0,
    transform=valid_kfold,
)

validation_frac = Requirement[float, float](
    name="validation_frac",
    description="What fraction of training data to split into validation data (or 0 to use test data as validation data)",
    default=0.0,
    bounds=(0, 0, 1.0),
)

data = [kfold, validation_frac]

# Training Settings
epochs_colour = Requirement[int, int](
    name="epochs_colour",
    description="Number of epochs in colour training",
    default=1000,
)
epochs_latent = Requirement[int, int](
    name="epochs_latent",
    description="Number of epochs in latent training",
    default=1000,
)
epochs_amplitude = Requirement[int, int](
    name="epochs_amplitude",
    description="Number of epochs in amplitude training",
    default=1000,
)
epochs_all = Requirement[int, int](
    name="epochs_all",
    description="Number of epochs in training all parameters",
    default=5000,
)

validate_every_n = Requirement[int, int](
    name="validate_every_n",
    description="How many steps between validation",
    default=100,
)

split_training = Requirement[bool, bool](
    name="split_training",
    description="Split training into several stages, otherwise train all parameters at once",
    default=True,
)

min_train_redshift = Requirement[float, float](
    name="min_train_redshift",
    description="Cut spectral data with redshift below this",
    default=0.02,
)

max_train_redshift = Requirement[float, float](
    name="max_train_redshift",
    description="Cut spectral data with redshift above this",
    default=1.0,
    transform=lambda train_redshift, _, opts: (True, train_redshift)
    if train_redshift > opts["MIN_TRAIN_REDSHIFT"]
    else (
        False,
        f"max_train_redshift={train_redshift} must be greater than min_train_redshift={opts['MIN_TRAIN_REDSHIFT']}",
    ),
)

min_train_phase = Requirement[int, int](
    name="min_train_phase",
    description="Cut spectral data with phase below this",
    default=-10,
)

max_train_phase = Requirement[int, int](
    name="max_train_phase",
    description="Cut spectral data with phase above this",
    default=40,
    transform=lambda train_phase, _, opts: (True, train_phase)
    if train_phase > opts["MIN_TRAIN_PHASE"]
    else (
        False,
        f"max_train_phase={train_phase} must be greater than min_train_phase={opts['MIN_TRAIN_PHASE']}",
    ),
)


def valid_colourlaw(file: str, cfg: "CFG", _2: "CFG"):
    if not file:
        return (True, None)
    path = Path(file)
    if not path.is_absolute():
        path = cfg["BASE"] / path
    path = path.resolve()
    if not path.exists():
        return (False, f"{path} does not exist")
    return (True, path)


colourlaw = Requirement[str, Path | None](
    name="colourlaw",
    description="Path to a colourlaw file",
    default="",
    transform=valid_colourlaw,
)

loss = Requirement[str, str](
    name="loss",
    description="Loss function",
    default="WHUBER",
    choice=[
        "MAE",
        "WMAE",
        "MSE",
        "WMSE",
        "RMSE",
        "WRMSE",
        "NGLL",
        "HUBER",
        "WHUBER",
        "MAGNITUDE",
    ],
)

batch_size = Requirement[int, int](
    name="batch_size",
    description="Size of each batch",
    default=57,
    transform=lambda batch, _1, _2: (True, batch)
    if batch > 0
    else (False, f"batch_size: {batch} is not strictly positive"),
)

noise_scale = Requirement[float, float](
    name="noise_scale",
    description="Scale of observation uncertainty to include in training",
    default=1.0,
)

mask_frac = Requirement[float, float](
    name="mask_frac",
    description="Fraction of spectra to randomly mask",
    default=0.1,
)

sigma_time = Requirement[float, float](
    name="sigma_time",
    description="sigma_time=n: Enforce a constant time uncertainty of n / 50. sigma_time=0: Enforce SALT-based time uncertainties (dphase / 50). sigma_time = -1: Do not enforce any time uncertainties",
    default=0.3,
)

training = [
    epochs_colour,
    epochs_latent,
    epochs_amplitude,
    epochs_all,
    validate_every_n,
    split_training,
    min_train_redshift,
    max_train_redshift,
    min_train_phase,
    max_train_phase,
    colourlaw,
    loss,
    batch_size,
    noise_scale,
    mask_frac,
    sigma_time,
]

# Network Settings
layer_type = Requirement[str, str](
    name="layer_type",
    description="Whether to create a DENSE or CONVOLUTIONAL layer",
    choice=["DENSE", "CONVOLUTIONAL"],
    default="DENSE",
)

activation = Requirement[str, str](
    name="activation",
    description="Network activation function",
    choice=["elu", "gelu", "relu", "swish", "tanh"],
    default="relu",
)

lr = Requirement[float, float](
    name="lr",
    description="Learning rate",
    default=0.005,
)
lr_deltat = Requirement[float, float](
    name="lr_deltat",
    description="Learning rate per unit time",
    default=0.001,
)
lr_decay_rate = Requirement[float, float](
    name="lr_decay_rate",
    description="Learning rate decay",
    default=0.95,
)
lr_decay_steps = Requirement[int, int](
    name="lr_decay_steps",
    description="Learning rate decay steps",
    default=300,
)
weight_decay_rate = Requirement[float, float](
    name="weight_decay_rate",
    description="Weighted optimiser decay rate",
    default=0.00001,
)

optimiser = Requirement[str, str](
    name="optimiser",
    description="Which optimiser to use",
    default="ADAMW",
    choice=["ADAM", "ADAMW"],
)
scheduler = Requirement[str, str](
    name="scheduler",
    description="Which scheduler to use (or empty for None)",
    default="EXPONENTIAL",
    choice=["", "EXPONENTIAL"],
)

kernel_regulariser = Requirement[float, float](
    name="kernel_regulariser",
    description="Value of L2 regularisation penalty (or 0 for no penalty)",
    default=0.0,
    transform=lambda n, _1, _2: (True, n)
    if n >= 0
    else (False, f"kernel_regulariser: {n} is negative"),
)

dropout = Requirement[float, float](
    name="dropout",
    description="Rate of dropout (or 0 for no dropout)",
    default=0.0,
    bounds=(0, 1),
)

batchnorm = Requirement[bool, bool](
    name="batchnorm",
    description="Whether to add a Batch Normalisation layer",
    default=False,
)

physical_latent = Requirement[bool, bool](
    name="physical_latent",
    description="Whether to include physical parameters in latent space",
    default=True,
)

network_settings = [
    layer_type,
    activation,
    lr,
    lr_deltat,
    lr_decay_rate,
    lr_decay_steps,
    weight_decay_rate,
    scheduler,
    optimiser,
    kernel_regulariser,
    dropout,
    batchnorm,
    physical_latent,
]

# Data Dimensions
cond_dim = Requirement[int, int](
    name="cond_dim",
    description="Dimension of conditional layer",
    default=1,
    transform=lambda dim, _1, _2: (True, dim)
    if dim > 0
    else (False, f"cond_dim: {dim} is not strictly positive"),
)


def valid_encode_dims(dims, _1: "CFG", _2: "CFG"):
    if len(dims) == 0:
        return (False, f"encode_dims: {dims} are empty")
    if any(dim <= 0 for dim in dims):
        return (
            False,
            f"encode_dims: {dims} contains elements which aren't strictly positive",
        )
    if not all(x > y for x, y in itertools.pairwise(dims)):
        return (False, f"encode_dims: {dims} is not monotonically decreasing")
    return (True, dims)


encode_dims = Requirement[list, list](
    name="encode_dims",
    description="Dimensions of each layer in the encoder. The final layer is automatically calculated to be n_timemax",
    default=[256, 128],
    transform=valid_encode_dims,
)


def valid_decode_dims(dims, _1: "CFG", params: "CFG"):
    if len(dims) == 0:
        dims = params["ENCODE_DIMS"][::-1]
    if any(dim <= 0 for dim in dims):
        return (
            False,
            f"decode_dims: {dims} contains elements which aren't strictly positive",
        )
    if not all(x < y for x, y in itertools.pairwise(dims)):
        return (False, f"decode_dims: {dims} is not monotonically increasing")
    return (True, dims)


decode_dims = Requirement[list, list](
    name="decode_dims",
    description="Dimensions of each layer in the decoder. Defaults to the reverse of the encode_dims",
    default=[],
    transform=valid_decode_dims,
)

latent_dim = Requirement[int, int](
    name="latent_dim",
    description="Number of latent variables",
    default=3,
)

dimensions = [cond_dim, encode_dims, decode_dims, latent_dim]

required: list["REQ"] = []
optional: list["REQ"] = []
required_params: list["REQ"] = []
optional_params: list["REQ"] = [*data, *training, *network_settings, *dimensions]
prev: list[str] = []
