from typing import TYPE_CHECKING
from pathlib import Path
import itertools

from suPAErnova.config.requirements import Requirement

if TYPE_CHECKING:
    from suPAErnova.utils.typing import CFG
    from suPAErnova.config.requirements import REQ


# Training Settings
epochs_physical = Requirement[int, int](
    name="epochs_physical",
    description="Number of epochs in physical parameter training",
    default=1000,
)
epochs_latent = Requirement[int, int](
    name="epochs_latent",
    description="Number of epochs in latent training",
    default=1000,
)
epochs_final = Requirement[int, int](
    name="epochs_final",
    description="Number of epochs in final training",
    default=5000,
)
validate_every_n = Requirement[int, int](
    name="validate_every_n",
    description="How many steps between validation",
    default=100,
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

training = [epochs_physical, epochs_latent, epochs_final, colourlaw]

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
    description="Learning rate decay percentage",
    default=0.95,
)
lr_decay_steps = Requirement[int, int](
    name="lr_decay_steps",
    description="Learning rate decay steps",
    default=300,
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
optional_params: list["REQ"] = [*training, *network_settings, *dimensions]
prev: list[str] = []
