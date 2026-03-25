import pytest
import numpy as np

from patch_denoise.simulation.activations import add_frames
from patch_denoise.simulation.noise import add_temporal_gaussian_noise
from patch_denoise.simulation.phantom import g_factor_map, mr_shepp_logan_t2_star


@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(42)


@pytest.fixture(scope="session")
def phantom(N_rep=20):
    """Create a dummy phantom with fake activations."""
    return add_frames(mr_shepp_logan_t2_star(64)[32], N_rep)


@pytest.fixture(scope="session")
def noisy_phantom(phantom, rng):
    """Create noisy version of phantom."""
    g_map = g_factor_map(phantom.shape[:-1])
    return add_temporal_gaussian_noise(
        phantom,
        sigma=1,
        rng=rng,
        g_factor_map=g_map,
    )
