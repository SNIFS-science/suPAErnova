from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from suPAErnova.steps.data import DataStep, SNPAEData


@pytest.fixture
def data_step_factory() -> "DataStep":
    pass


@pytest.fixture
def data_result_factory() -> "SNPAEData":
    pass
