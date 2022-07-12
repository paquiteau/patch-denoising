import pytest
import numpy as np

from numpy.testing import assert_allclose, assert_almost_equal

from denoiser.space_time.utils import (
    svd_analysis,
    svd_synthesis,
    eig_analysis,
    eig_synthesis,
    marshenko_pastur_median,
    estimate_noise,
)

from itertools import product


parametrize_random_matrix = pytest.mark.parametrize(
    "matrix",
    [
        {"cplx": a, "shape": b, "sigma": c}
        for a, b, c in product([False], [(100, 50), (100, 10)], [0.1, 1, 10])
    ],
    indirect=True,
)


@pytest.fixture()
def matrix(request):
    rng = np.random.RandomState(42)
    sigma = request.param["sigma"]
    M, N = request.param["shape"]
    if not request.param["cplx"]:
        return rng.randn(M, N) * sigma
    return rng.randn(M, N) * sigma + 1j * rng.randn(M, N) * sigma

@pytest.fixture(scope="module")
def rng():
    return np.random.RandomState(42)

@pytest.mark.parametrize("beta", np.arange(1, 10) * 0.1)
def test_marshenko_pastur_median(beta, rng,  n_runs=10000, n_samples=1000):
    """Test the median estimation of Marshenko Pastur law"""
    print(beta)
    beta_p = (1 + np.sqrt(beta)) ** 2
    beta_m = (1 - np.sqrt(beta)) ** 2

    def f(x):
        """Marchenko Pastur Probability density function"""
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


@pytest.fixture(scope="function")
def medium_random_matrix(rng):
    shape=(200, 200, 100)
    return rng.randn(*shape)

@pytest.mark.parametrize("block_dim", range(1,6))
def test_noise_estimation(medium_random_matrix, block_dim):

    noise_map = estimate_noise(medium_random_matrix, block_dim)

    real_std = np.nanstd(medium_random_matrix)

    assert np.nanstd(noise_map) <= 0.2 * real_std

@parametrize_random_matrix
def test_svd(matrix):
    U, S, V, M  = svd_analysis(matrix)
    new_matrix = svd_synthesis(U, S, V, M, idx=len(S))

    # TODO Refine the precision criteria
    assert np.sqrt(np.mean(np.square(matrix - new_matrix)))  <= np.std(matrix)/10

@parametrize_random_matrix
def test_eig(matrix):
    A, d, W, M  = eig_analysis(matrix)
    new_matrix = eig_synthesis(A, W, M, max_val=len(M))
    # TODO Refine the precision criteria
    assert np.sqrt(np.mean(np.square(matrix - new_matrix)))  <= np.std(matrix)
