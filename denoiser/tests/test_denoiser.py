import pytest
import numpy as np


from denoiser.simulation.phantom import g_factor_map, mr_shepp_logan_t2_star
from denoiser.simulation.noise import add_temporal_gaussian_noise
from denoiser.simulation.activations import add_activations

from denoiser.denoise import mp_pca, hybrid_pca, nordic, optimal_thresholding, raw_svt


@pytest.fixture(scope="module")
def phantom(N_rep=20):
    return add_activations(mr_shepp_logan_t2_star(64)[32], N_rep)


@pytest.fixture(scope="module")
def noisy_phantom(phantom, rng):
    m = np.mean(phantom)
    g_map = g_factor_map(phantom.shape[:-1])
    return add_temporal_gaussian_noise(
        phantom,
        sigma=1,
        rng=rng,
        g_factor_map=g_map,
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
@pytest.mark.parametrize("loss", ["fro", "nuc", "ope"])
def test_optimal_denoiser(phantom, noisy_phantom, recombination, loss):
    """Test the Optimal Thresholding denoiser"""
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
