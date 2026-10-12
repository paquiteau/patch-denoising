"""Test space-times utilities."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import cumulative_trapezoid

from patch_denoise.space_time.base import (
    PatchedArray,
    center_padding,
    center_time_slice,
)
from patch_denoise.space_time.utils import (
    eig_analysis,
    eig_synthesis,
    estimate_noise,
    marchenko_pastur_median,
    patch_starts,
    svd_analysis,
    svd_synthesis,
)


@pytest.fixture(params=[(100, 50), (100, 10)], ids=str)
def matrix(request):
    """Random matrix of a given shape."""
    return np.random.default_rng(42).standard_normal(request.param)


@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_marchenko_pastur_median(beta):
    """The median matches the one of the tabulated Marchenko-Pastur density."""
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    x = np.linspace(lo, hi, 200001)
    pdf = np.sqrt(np.clip((hi - x) * (x - lo), 0, None)) / (2 * np.pi * x * beta)
    cdf = cumulative_trapezoid(pdf, x, initial=0)
    expected = np.interp(0.5, cdf / cdf[-1], x)
    assert marchenko_pastur_median(beta, eps=1e-7) == pytest.approx(expected, rel=1e-5)


def test_noise_estimation():
    """The patch-wise standard deviation of white noise is close to 1."""
    noise_map = estimate_noise(np.random.default_rng(0).normal(size=(40, 40, 100)), 5)
    # Only voxels at the center of a patch are estimated.
    npt.assert_allclose(noise_map[2:-2, 2:-2], 1, rtol=0.1)


def test_svd(matrix):
    """Test SVD functions."""
    U, S, V, M = svd_analysis(matrix)
    new_matrix = svd_synthesis(U, S, V, M, idx=len(S))

    # TODO Refine the precision criteria
    assert np.sqrt(np.mean(np.square(matrix - new_matrix))) <= np.std(matrix) / 10


def test_eig(matrix):
    """Test SVD via eigenvalue decomposition."""
    A, d, W, M = eig_analysis(matrix)
    new_matrix = eig_synthesis(A, W, M, max_val=W.shape[1])
    # TODO Refine the precision criteria
    assert np.sqrt(np.mean(np.square(matrix - new_matrix))) <= np.std(matrix)


@pytest.mark.parametrize(
    "data_size, patch_size, step, expected",
    [
        (10, 4, 3, [0, 3, 6]),  # exact fit
        (11, 4, 3, [0, 3, 6, 7]),  # remainder: last patch on the border
        (8, 8, 1, [0]),  # patch spans the axis
        (5, 8, 1, [0]),  # patch larger than the axis
    ],
)
def test_patch_starts(data_size, patch_size, step, expected):
    """Patches are strided, plus one on the border if the stride leaves a gap."""
    npt.assert_array_equal(patch_starts(data_size, patch_size, step), expected)


@pytest.mark.parametrize("patch_overlap", [(0, 0, 0), (2, 1, 0), (4, 4, 0)])
def test_patched_array_covers_every_voxel(patch_overlap):
    """Every voxel is in a patch, whatever the stride."""
    shape, patch_shape = (13, 11, 6), (5, 5, 6)
    counts = PatchedArray(shape, patch_shape, patch_overlap, dtype=np.int32)
    for i in range(counts.n_patches):
        counts.add2patch(i, 1)
    assert np.all(counts._arr > 0)
    # The last patch along each axis ends on the border.
    last = counts.idx2slice(counts.n_patches - 1)
    assert tuple(sl.stop for sl in last) == shape


def test_center_padding():
    """Only the border margin reached by the mask is padded."""
    patch_shape, data_shape = (5, 4, 6), (20, 20, 6)
    # No mask: the full margin, half a patch on each side; the last axis is
    # spanned by the patch and never padded.
    assert center_padding(patch_shape, data_shape) == [(2, 2), (2, 1), (0, 0)]

    mask = np.zeros(data_shape[:-1], dtype=bool)
    mask[5:15, 5:15] = True  # away from the borders: nothing to pad
    assert center_padding(patch_shape, data_shape, mask) == [(0, 0)] * 3

    mask[1, 10] = True  # one voxel from the low border of axis 0
    assert center_padding(patch_shape, data_shape, mask) == [(1, 0), (0, 0), (0, 0)]

    # A time window shorter than the data is not padded either, unlike space.
    assert center_padding((5, 4, 3), data_shape) == [(2, 2), (2, 1), (0, 0)]


def test_center_time_slice_covers_every_frame_once():
    """With maximal overlap, the kept frames of all windows tile the time axis."""
    patch_t, data_t = 5, 12
    covered = np.zeros(data_t, dtype=int)
    for start in range(data_t - patch_t + 1):
        covered[start : start + patch_t][center_time_slice(start, patch_t, data_t)] += 1
    assert np.all(covered == 1)
    # A window spanning the whole time axis keeps it entirely.
    assert center_time_slice(0, data_t, data_t) == slice(0, data_t)
