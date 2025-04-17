from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from suPAErnova.steps.pae.pae import PAEStep


@pytest.mark.setup
def test_setup(paestep: "PAEStep") -> None:
    pass
