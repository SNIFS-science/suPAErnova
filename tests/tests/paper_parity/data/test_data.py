import numpy as np
import pytest

from suPAErnova.steps.data import SNPAEData

KEYS = list(SNPAEData.model_fields.keys())

# def array_diff(
#     snpae: "npt.NDArray[Any]",
#     legacy: "npt.NDArray[Any]",
#     *,
#     exact: bool,
#     max_diffs: int = 10,
#     data: SNPAEData,
#     data_legacy: SNPAEData,
# ) -> str:
#     # Element-wise comparison
#     diff_mask = snpae != legacy if exact else np.logical_not(np.isclose(snpae, legacy))
#     diff_indices = np.argwhere(diff_mask)
#
#     rtn = f"Arrays differ at {len(diff_indices)} / {snpae.size} ({int(100 * len(diff_indices) / snpae.size)}%) positions (showing up to {max_diffs}):"
#     for idx in diff_indices[:max_diffs]:
#         a_val = snpae[tuple(idx)]
#         b_val = legacy[tuple(idx)]
#         id = data.spectra_id[idx[0], idx[1], 0]
#         id_legacy = data_legacy.spectra_id[idx[0], idx[1], 0]
#         rtn += f"\n  At index {tuple(int(i) for i in idx)}: SuPAErnova = {a_val}, Legacy = {b_val}"
#         rtn += f"\n      (SuPAErnova id: {id}, Legacy id: {id_legacy})"
#
#     if len(diff_indices) > max_diffs:
#         rtn += f"… and {len(diff_indices) - max_diffs} more differences."
#     return rtn


@pytest.mark.parametrize("key", KEYS)
def test_shapes(key: str, snpae_data: "SNPAEData", legacy_data: "SNPAEData") -> None:
    snpae_shape = getattr(snpae_data, key).shape
    legacy_shape = getattr(legacy_data, key).shape
    assert snpae_shape == legacy_shape


@pytest.mark.parametrize("key", KEYS)
def test_matching_values(
    key: str, snpae_data: "SNPAEData", legacy_data: "SNPAEData"
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
