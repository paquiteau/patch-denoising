"""Tests for the GPU denoising pipeline."""

import numpy as np
import numpy.testing as npt
import pytest

from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.denoise import mp_pca, optimal_thresholding
from patch_denoise.simulation.activations import add_frames
from patch_denoise.simulation.noise import add_temporal_gaussian_noise
from patch_denoise.simulation.phantom import g_factor_map, mr_shepp_logan_t2_star

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")

PATCH_SHAPE = (6, 6, 6, 8)
PATCH_OVERLAP = (3, 3, 3, 0)
GPU_METHODS = ["mp-pca", "optimal-fro", "optimal-nuc", "optimal-ope"]


def _method_kwargs(method):
    """Extra denoising kwargs that only apply to a given method."""
    return {"threshold_scale": 2.3} if method == "mp-pca" else {}


@pytest.fixture(scope="module")
def noisy_phantom_4d(rng):
    """Create a small 4D (X, Y, Z, T) noisy phantom, real and imaginary parts."""
    phantom = add_frames(mr_shepp_logan_t2_star(24), 8)
    g_map = g_factor_map(phantom.shape[:-1])
    real = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    imag = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    return phantom, real, imag


@pytest.mark.parametrize("method", GPU_METHODS)
def test_main_gpu_complex64(noisy_phantom_4d, method):
    from patch_denoise.gpu.main import main_gpu

    phantom, real, imag = noisy_phantom_4d
    data = (real + 1j * imag).astype(np.complex64)
    mask = np.ones(data.shape[:-1], dtype=bool)

    denoised, weights, noise_std_map, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method=method,
        mask=mask,
        batch_size=8,
        compile=False,
        **_method_kwargs(method),
    )

    assert denoised.shape == data.shape
    assert denoised.dtype == np.complex64
    assert np.all(np.isfinite(denoised))

    # a correctly-sized accumulation buffer should actually denoise the phantom
    noise_std_before = np.sqrt(np.nanmean(np.nanvar(np.abs(data - phantom), axis=-1)))
    noise_std_after = np.sqrt(
        np.nanmean(np.nanvar(np.abs(denoised - phantom), axis=-1))
    )
    assert noise_std_after < noise_std_before


@pytest.mark.parametrize("method", GPU_METHODS)
def test_main_gpu_matches_cpu(noisy_phantom_4d, method):
    """GPU denoising should match the CPU reference implementation.

    Uses a batch size that does not divide the number of temporal features
    (8), so a regression of the per-batch-row rank-selection bugs would
    show up as a crash rather than just a numerical mismatch.
    """
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)

    if method == "mp-pca":
        cpu_denoised, _, cpu_noise_std, _ = mp_pca(
            data,
            patch_shape=PATCH_SHAPE,
            patch_overlap=PATCH_OVERLAP,
            mask=mask,
            recombination="weighted",
            threshold_scale=2.3,
        )
    else:
        cpu_denoised, _, cpu_noise_std, _ = optimal_thresholding(
            data,
            patch_shape=PATCH_SHAPE,
            patch_overlap=PATCH_OVERLAP,
            mask=mask,
            recombination="weighted",
            loss=method.split("-")[-1],
        )

    gpu_denoised, _, gpu_noise_std, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method=method,
        mask=mask,
        batch_size=5,
        compile=False,
        **_method_kwargs(method),
    )

    npt.assert_allclose(gpu_denoised, cpu_denoised, rtol=1e-5, atol=1e-2)

    # Both backends should report a standard-deviation map, at the full
    # (X, Y, Z, T) shape of the input (not a variance, and not reduced
    # over any axis).
    assert gpu_noise_std.shape == cpu_noise_std.shape == data.shape
    npt.assert_allclose(gpu_noise_std, cpu_noise_std, rtol=1e-5, atol=1e-2)
