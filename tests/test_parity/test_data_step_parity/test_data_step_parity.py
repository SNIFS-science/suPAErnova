from typing import TYPE_CHECKING

import numpy as np
import pytest

from suPAErnova.steps.data import SNPAEData

if TYPE_CHECKING:
    from typing import Any

    from numpy import typing as npt


KEYS = list(SNPAEData.model_fields.keys())


def array_diff(
    snpae: "npt.NDArray[Any]",
    legacy: "npt.NDArray[Any]",
    *,
    exact: bool,
    max_diffs: int = 10,
    data: SNPAEData,
    data_legacy: SNPAEData,
) -> str:
    # Element-wise comparison
    diff_mask = snpae != legacy if exact else np.logical_not(np.isclose(snpae, legacy))
    diff_indices = np.argwhere(diff_mask)

    rtn = f"Arrays differ at {len(diff_indices)} / {snpae.size} ({int(100 * len(diff_indices) / snpae.size)}%) positions (showing up to {max_diffs}):"
    for idx in diff_indices[:max_diffs]:
        a_val = snpae[tuple(idx)]
        b_val = legacy[tuple(idx)]
        id = data.spectra_id[idx[0], idx[1], 0]
        id_legacy = data_legacy.spectra_id[idx[0], idx[1], 0]
        rtn += f"\n  At index {tuple(int(i) for i in idx)}: SuPAErnova = {a_val}, Legacy = {b_val}"
        rtn += f"\n      (SuPAErnova id: {id}, Legacy id: {id_legacy})"

    if len(diff_indices) > max_diffs:
        rtn += f"… and {len(diff_indices) - max_diffs} more differences."
    return rtn


@pytest.mark.setup
def test_setup(data: "SNPAEData", data_legacy: "SNPAEData") -> None:
    pass


@pytest.mark.parametrize("key", KEYS)
def test_shapes(key: str, data: "SNPAEData", data_legacy: "SNPAEData") -> None:
    shape = getattr(data, key).shape
    shape_legacy = getattr(data_legacy, key).shape
    assert shape == shape_legacy, (
        f"Shape of {key} data for suPAErnova doesn't match shape of {key} data for suPAErnova_legacy.\n{shape}!={shape_legacy}"
    )


@pytest.mark.parametrize("key", KEYS)
def test_matching_values(key: str, data: "SNPAEData", data_legacy: "SNPAEData") -> None:
    vals = getattr(data, key)
    vals_legacy = getattr(data_legacy, key)

    if np.issubdtype(vals.dtype, np.number) and np.issubdtype(
        vals_legacy.dtype, np.number
    ):
        assert np.allclose(vals, vals_legacy), (
            f"{key} for suPAErnova doesn't match {key} for suPAErnova_legacy.\n{array_diff(vals, vals_legacy, exact=False, data=data, data_legacy=data_legacy)}"
        )
    else:
        assert np.array_equal(vals, vals_legacy), (
            f"{key} for suPAErnova doesn't match {key} for suPAErnova_legacy.\n{array_diff(vals, vals_legacy, exact=True, data=data, data_legacy=data_legacy)}"
        )
