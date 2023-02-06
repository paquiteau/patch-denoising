"""Functions to add different source of temporal noise to data."""
import numpy as np
from numpy.random import default_rng


def add_temporal_gaussian_noise(array, sigma=1, g_factor_map=None, rng=None):
    """Add gaussian noise to array.

    Parameters
    ----------
    array: numpy.ndarray
        The noise_free ND-array, where the last dimension is a dynamical one
        (e.g. time)
    sigma: float
        gaussian noise variance
    g_factor_map: numpy.ndarray, optional
        Spatial variation of the noise ((N-1)D array). default is identity.

    Returns
    -------
    numpy.ndarray
        A noisy array
    """
    if rng is None:
        rng = default_rng()
    shape = array.shape

    g_noise = sigma * rng.standard_normal(shape)
    if g_factor_map is None:
        g_factor_map = np.ones(shape[:-1])

    if np.iscomplex(array).any():
        g_noise += 1j * sigma * rng.standard_normal(shape)
    return array + (g_noise * g_factor_map[..., None])


def add_temporal_rician_noise(array, scale=1, rng=None):
    """Add center rician noise to array.

    Parameters
    ----------
    array: numpy.ndarray
        The noise-free array
    sigma: float
        The scale of the Rice distribution

    Notes
    -----
    This function considered centered Rician noise [1]_,
    and thus the noise generated follows a Rayleigh distribution [2]_.

    References
    ----------
    .. [1] https://en.m.wikipedia.org/wiki/Rice_distribution
    .. [2] https://en.m.wikipedia.org/wiki/Rayleigh_distribution
    """
    if rng is None:
        rng = default_rng()
    noise = rng.rayleigh(scale, array.shape)

    return array + noise
