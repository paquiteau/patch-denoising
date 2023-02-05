"""Test space-times utilities."""
from itertools import product

import numpy as np
import pytest

from patch_denoise.space_time.utils import (
    eig_analysis,
    eig_synthesis,
    estimate_noise,
    marshenko_pastur_median,
    svd_analysis,
    svd_synthesis,
)

parametrize_random_matrix = pytest.mark.parametrize(
    "matrix",
    [
        {"cplx": a, "shape": b, "sigma": c}
        for a, b, c in product([False], [(100, 50), (100, 10)], [0.1, 1, 10])
    ],
    indirect=True,
)


@pytest.fixture(scope="function")
def medium_random_matrix(rng):
    """Create random 3D array. with size (200, 200, 100)."""
    shape = (200, 200, 100)
    return rng.randn(*shape)


@pytest.fixture()
def matrix(request):
    """Create random matrix on command with shape and noise level."""
    rng = np.random.RandomState(42)
    sigma = request.param["sigma"]
    M, N = request.param["shape"]
    if not request.param["cplx"]:
        return rng.randn(M, N) * sigma
    return rng.randn(M, N) * sigma + 1j * rng.randn(M, N) * sigma


@pytest.mark.parametrize("beta", np.arange(1, 10) * 0.1)
def test_marshenko_pastur_median(beta, rng, n_runs=10000, n_samples=1000):
    """Test the median estimation of Marshenko Pastur law."""
    print(beta)
    beta_p = (1 + np.sqrt(beta)) ** 2
    beta_m = (1 - np.sqrt(beta)) ** 2

    def f(x):
        """Marchenko Pastur Probability density function."""
        if beta_p >= x >= beta_m:
            return np.sqrt((beta_p - x) * (x - beta_m)) / (2 * np.pi * x * beta)
        else:
            return 0

    integral_median = marshenko_pastur_median(beta, eps=1e-7)

    vals = np.linspace(beta_m, beta_p, n_samples)
    proba = np.array(list(map(f, vals)))
    proba /= np.sum(proba)
    samples = np.zeros(n_runs)
    for i in range(n_runs):
        samples[i] = np.median(rng.choice(vals, size=n_runs, p=proba))
    #    montecarlo_median = np.mean(samples)

    # TODO: increase precision of montecarlo simulation
    assert np.std(samples) <= 0.1 * integral_median


@pytest.mark.parametrize("block_dim", range(5, 10))
def test_noise_estimation(medium_random_matrix, block_dim):
    """Test noise estimation."""
    noise_map = estimate_noise(medium_random_matrix, block_dim)

    real_std = np.nanstd(medium_random_matrix)
    err = np.nanmean(noise_map - real_std)
    assert err <= 0.1 * real_std


@parametrize_random_matrix
def test_svd(matrix):
    """Test SVD functions."""
    U, S, V, M = svd_analysis(matrix)
    new_matrix = svd_synthesis(U, S, V, M, idx=len(S))

    # TODO Refine the precision criteria
    assert np.sqrt(np.mean(np.square(matrix - new_matrix))) <= np.std(matrix) / 10


@parametrize_random_matrix
def test_eig(matrix):
    """Test SVD via eigenvalue decomposition."""
    A, d, W, M = eig_analysis(matrix)
    new_matrix = eig_synthesis(A, W, M, max_val=len(M))
    # TODO Refine the precision criteria
    assert np.sqrt(np.mean(np.square(matrix - new_matrix))) <= np.std(matrix)
