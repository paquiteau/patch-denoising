"""Test for the different denoising methods."""

import numpy as np
import numpy.testing as npt
import pytest

from patch_denoise.denoise import (
    adaptive_thresholding,
    hybrid_pca,
    mp_pca,
    nordic,
    optimal_thresholding,
    raw_svt,
)
from patch_denoise.simulation.phantom import g_factor_map


def _noise_std(data, phantom):
    return np.sqrt(np.nanmean(np.nanvar(data - phantom, axis=-1)))


@pytest.mark.parametrize(
    "denoise_func, kwargs",
    [
        pytest.param(mp_pca, {"threshold_scale": 2.3}, id="mp-pca"),
        pytest.param(hybrid_pca, {"noise_std": 1.0}, id="hybrid-pca"),
        pytest.param(nordic, {"noise_std": 1.0}, id="nordic"),
        pytest.param(raw_svt, {"threshold": 10}, id="raw-svt"),
        *(
            pytest.param(optimal_thresholding, {"loss": loss}, id=f"optimal-{loss}")
            for loss in ["fro", "nuc", "ope"]
        ),
    ],
)
def test_denoiser(phantom, noisy_phantom, denoise_func, kwargs):
    """Every denoiser reduces the noise."""
    denoised, *_ = denoise_func(noisy_phantom, patch_shape=6, patch_overlap=5, **kwargs)
    assert _noise_std(denoised, phantom) < _noise_std(noisy_phantom, phantom)


@pytest.mark.parametrize(
    "denoise_func, kwargs",
    [
        pytest.param(
            optimal_thresholding,
            {"patch_overlap": 9, "loss": "fro"},
            id="optimal-fro",
        ),
        *(
            pytest.param(
                adaptive_thresholding,
                {"patch_overlap": 0, "method": method, "gamma0": gamma, "nbsim": 500},
                id=f"adaptive-{method}",
            )
            for method, gamma in [
                ("qut", None),
                ("gsure", np.linspace(1, 5, 10)),
                ("sure", np.linspace(1, 5, 10)),
            ]
        ),
    ],
)
def test_denoiser_noise_map(phantom, noisy_phantom, denoise_func, kwargs):
    """Denoisers given a noise std map reduce the noise."""
    denoised, *_ = denoise_func(
        noisy_phantom,
        patch_shape=10,
        noise_std=2 * g_factor_map(phantom.shape[:-1]),
        **kwargs,
    )
    assert _noise_std(denoised, phantom) < _noise_std(noisy_phantom, phantom)


# 'weighted' is the default, tested above.
@pytest.mark.parametrize(
    "recombination, patch_shape, patch_overlap",
    [
        ("average", 6, 5),
        ("center", 6, 5),
        # Time axis spanning the whole data extent (-1 in the CLI).
        ("center", (6, 6, -1), (5, 5, -1)),
    ],
)
def test_recombination(
    phantom, noisy_phantom, recombination, patch_shape, patch_overlap
):
    """Every recombination denoises every volume.

    Regression test: with 'center' and a full-extent time axis, the time
    profile at each spatial patch center must be kept whole, not collapsed to
    a single time point -- which used to leave every other volume at 0.
    """
    denoised, *_ = mp_pca(
        noisy_phantom,
        patch_shape=patch_shape,
        patch_overlap=patch_overlap,
        threshold_scale=2.3,
        recombination=recombination,
    )
    assert np.all(np.isfinite(denoised))
    assert np.all(np.count_nonzero(denoised, axis=(0, 1)) > 0)
    assert _noise_std(denoised, phantom) < _noise_std(noisy_phantom, phantom)


def test_center_requires_maximal_overlap(noisy_phantom):
    """'center' recombination keeps one voxel per patch: other overlaps leave holes."""
    with pytest.raises(ValueError, match="maximal overlap"):
        mp_pca(noisy_phantom, patch_shape=6, patch_overlap=3, recombination="center")


def test_center_padding_with_noise_map(noisy_phantom):
    """'center' recombination reaches the border, with a mask and noise std map too.

    The mask and noise map are padded along with the data, before the denoiser
    builds its per-patch noise variances from it.
    """
    data = noisy_phantom[:16, :16, :8]  # only the border matters here
    denoised, _, noise, _ = hybrid_pca(
        data,
        patch_shape=6,
        patch_overlap=5,
        noise_std=g_factor_map(data.shape[:-1]),
        recombination="center",
        mask=np.ones(data.shape[:-1], dtype=bool),
    )
    assert denoised.shape == noise.shape == data.shape
    assert np.all(denoised != 0)


def test_raise_nordic(noisy_phantom):
    """Test raise error for nordic."""
    npt.assert_raises(
        ValueError,
        nordic,
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        noise_std="not_a_float_or_array",
    )
