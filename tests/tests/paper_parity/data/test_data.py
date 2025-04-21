import numpy as np
import pytest

from suPAErnova.configs.steps.data import DataStepResult

pytestmark = pytest.mark.data

KEYS = list(DataStepResult.model_fields.keys())


@pytest.mark.setup("snpae")
def test_snpae_data_setup(snpae_data: "DataStepResult") -> None:
    pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_data_setup(legacy_data: "DataStepResult") -> None:
    pass


@pytest.mark.parametrize("key", KEYS)
def test_shapes(
    key: str, snpae_data: "DataStepResult", legacy_data: "DataStepResult"
) -> None:
    snpae_shape = getattr(snpae_data, key).shape
    legacy_shape = getattr(legacy_data, key).shape
    assert snpae_shape == legacy_shape


@pytest.mark.parametrize("key", KEYS)
def test_matching_values(
    key: str, snpae_data: "DataStepResult", legacy_data: "DataStepResult"
) -> None:
    snpae_vals = getattr(snpae_data, key)
    legacy_vals = getattr(legacy_data, key)

    compare = (
        np.allclose
        if np.issubdtype(snpae_vals.dtype, np.number)
        and np.issubdtype(legacy_vals.dtype, np.number)
        else np.array_equal
    )
    assert compare(snpae_vals, legacy_vals)
