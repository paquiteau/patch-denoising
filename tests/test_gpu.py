"""Tests for the GPU denoising pipeline."""

import numpy as np
import pytest

from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.simulation.activations import add_frames
from patch_denoise.simulation.noise import add_temporal_gaussian_noise
from patch_denoise.simulation.phantom import g_factor_map, mr_shepp_logan_t2_star

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")


@pytest.fixture(scope="module")
def noisy_phantom_4d(rng):
    """Create a small 4D (X, Y, Z, T) noisy phantom, real and imaginary parts."""
    phantom = add_frames(mr_shepp_logan_t2_star(24), 8)
    g_map = g_factor_map(phantom.shape[:-1])
    real = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    imag = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    return phantom, real, imag


@pytest.mark.parametrize("method", ["optimal-fro", "optimal-nuc", "optimal-ope"])
def test_main_gpu_complex64(noisy_phantom_4d, method):
    from patch_denoise.gpu.main import main_gpu

    phantom, real, imag = noisy_phantom_4d
    data = (real + 1j * imag).astype(np.complex64)
    mask = np.ones(data.shape[:-1], dtype=bool)

    denoised, weights, noise_std_map, _ = main_gpu(
        data,
        patch_shape=(6, 6, 6, 8),
        patch_overlap=(3, 3, 3, 0),
        mask_threshold=50,
        recombination="weighted",
        method=method,
        mask=mask,
        batch_size=8,
        compile=False,
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
