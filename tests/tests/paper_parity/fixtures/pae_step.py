from typing import TYPE_CHECKING

import pytest

from suPAErnova.steps.pae.pae import PAEStep
from suPAErnova.configs.steps.pae.pae import PAEStepConfig

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.fixtures import SubRequest

    from suPAErnova.steps.data import DataStep


@pytest.fixture(scope="session")
def paestep(
    request: "SubRequest",
    datastep: "DataStep",
    root_path: "Path",
    data_path: "Path",
    cache_path: "Path",
    verbosity: int,
    *,
    force: bool,
) -> "None":
    # ((min_phase, max_phase), (_min_redshift, _max_redshift), train_frac, seed) = (
    #     request.param
    # )

    model_config = {}
    config = {
        "config": datastep.config,
        "paths": datastep.paths,
        "data": datastep.options.model_dump(),
        "pae": {"model": model_config},
    }
    return None
    pae = PAEStep(PAEStepConfig.from_config(config))
    assert pae is not None, "Error running PAEStep"
    return pae
