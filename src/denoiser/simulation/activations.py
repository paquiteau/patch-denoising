"""Create Dynamical Data from phantom."""

import numpy as np


def add_frames(volume, n_frames):
    """Add Activation to ground truth volume.

    Parameters
    ----------
    volume: numpy.ndarray
        The Static volume to augment
    n_frames: int
        The number of temporal frames to create

    Returns
    -------
    numpy.ndarray
        The temporal sequence of volume with activations.
    """
    return np.repeat(volume[..., np.newaxis], n_frames, axis=-1)
