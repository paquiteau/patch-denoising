import pytest
import numpy as np

from numpy.testing import assert_allclose, assert_almost_equal


from denoiser.simulation.phantom import mr_shepp_logan_t2_star
from denoiser.simulation.noise import add_temporal_gaussian_noise

from denoiser.denoise import mp_pca, hybrid_pca, nordic, optimal_thresholding, raw_svt


@pytest.fixture(scope="module")
def phantom(N_rep=20):
    return np.repeat(mr_shepp_logan_t2_star(64)[32, :, :, np.newaxis], N_rep, axis=-1)


@pytest.fixture(scope="module")
def noisy_phantom(phantom, rng):
    m = np.mean(phantom)
    return add_temporal_gaussian_noise(
        phantom,
        sigma=1,
        rng=rng,
    )


@pytest.mark.parametrize("recombination", ["weighted", "average", "center"])
def test_mppca_denoiser(phantom, noisy_phantom, recombination):
    """Test the MP-PCA denoiser"""
    print(noisy_phantom.shape)
    denoised, weights, noise = mp_pca(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        threshold_scale=2.3,
        recombination=recombination,
    )
    noise_std_before = np.sqrt(np.nanmean(np.nanvar(noisy_phantom - phantom, axis=-1)))
    noise_std_after = np.sqrt(np.nanmean(np.nanvar(denoised - phantom, axis=-1)))
    assert noise_std_after < noise_std_before


@pytest.mark.parametrize("recombination", ["weighted", "average", "center"])
def test_hybridpca_denoiser(phantom, noisy_phantom, recombination):
    """Test the Hybrid-PCA denoiser"""
    print(noisy_phantom.shape)
    denoised, weights, noise = hybrid_pca(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        noise_std=1.0,
        recombination=recombination,
    )

    noise_std_before = np.sqrt(np.nanmean(np.nanvar(noisy_phantom - phantom, axis=-1)))
    noise_std_after = np.sqrt(np.nanmean(np.nanvar(denoised - phantom, axis=-1)))
    assert noise_std_after < noise_std_before


@pytest.mark.parametrize("recombination", ["weighted", "average", "center"])
def test_nordic_denoiser(phantom, noisy_phantom, recombination):
    """Test the Hybrid-PCA denoiser"""
    print(noisy_phantom.shape)
    denoised, weights, noise = nordic(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        noise_std=1.0,
        recombination=recombination,
    )

    noise_std_before = np.sqrt(np.nanmean(np.nanvar(noisy_phantom - phantom, axis=-1)))
    noise_std_after = np.sqrt(np.nanmean(np.nanvar(denoised - phantom, axis=-1)))
    assert noise_std_after < noise_std_before


@pytest.mark.parametrize("recombination", ["weighted", "average", "center"])
def test_rawsvt_denoiser(phantom, noisy_phantom, recombination):
    """Test the Hybrid-PCA denoiser"""
    print(noisy_phantom.shape)
    denoised, weights, noise = raw_svt(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        noise_std=1.0,
        recombination=recombination,
    )

    noise_std_before = (
        np.sqrt(np.nanmean(np.nanvar(noisy_phantom - phantom, axis=-1))),
    )

    noise_std_after = np.sqrt(np.nanmean(np.nanvar(denoised - phantom, axis=-1)))
    assert noise_std_after < noise_std_before


@pytest.mark.parametrize("recombination", ["weighted", "average", "center"])
@pytest.mark.parametrize("loss", ["fro", "nuc", "op"])
def test_optimal_denoiser(phantom, noisy_phantom, recombination, loss):
    """Test the Hybrid-PCA denoiser"""
    print(noisy_phantom.shape)
    denoised, weights, noise = optimal_thresholding(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        recombination=recombination,
        loss=loss,
    )

    noise_std_before = np.sqrt(np.nanmean(np.nanvar(noisy_phantom - phantom, axis=-1)))
    noise_std_after = np.sqrt(np.nanmean(np.nanvar(denoised - phantom, axis=-1)))
    assert noise_std_after < noise_std_before * 1.1
