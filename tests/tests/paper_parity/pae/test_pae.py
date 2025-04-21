import numpy as np
import pytest

from suPAErnova.configs.steps.pae import PAEStepResult

pytestmark = pytest.mark.pae

KEYS = list(PAEStepResult.model_fields.keys())


@pytest.mark.setup("snpae")
def test_snpae_pae_setup(snpae_pae: "PAEStepResult") -> None:
    pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_pae_setup(legacy_pae: "PAEStepResult") -> None:
    pass


@pytest.mark.parametrize("key", KEYS)
def test_shapes(
    key: str, snpae_pae: "PAEStepResult", legacy_pae: "PAEStepResult"
) -> None:
    snpae_shape = getattr(snpae_pae, key).shape
    legacy_shape = getattr(legacy_pae, key).shape
    assert snpae_shape == legacy_shape


@pytest.mark.parametrize("key", KEYS)
def test_matching_values(
    key: str, snpae_pae: "PAEStepResult", legacy_pae: "PAEStepResult"
) -> None:
    snpae_vals = getattr(snpae_pae, key)
    legacy_vals = getattr(legacy_pae, key)

    compare = (
        np.allclose
        if np.issubdtype(snpae_vals.dtype, np.number)
        and np.issubdtype(legacy_vals.dtype, np.number)
        else np.array_equal
    )
    assert compare(snpae_vals, legacy_vals)
