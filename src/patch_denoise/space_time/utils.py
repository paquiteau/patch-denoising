"""Utilities for space-time denoising."""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.linalg import eigh, svd


def patch_starts(data_size: int, patch_size: int, step: int) -> NDArray:
    """Start index of every patch along one axis.

    Patches are placed every ``step`` voxels. When that leaves a remainder
    at the end of the axis, one last patch is aligned on the border, so that
    every voxel is covered. A patch spanning the whole axis gives ``[0]``.
    """
    last = data_size - patch_size
    if last <= 0:  # patch dimension cover the whole axis
        return np.zeros(1, dtype=np.int64)
    starts = np.arange(0, last + 1, step, dtype=np.int64)
    if starts[-1] != last:
        starts = np.append(starts, last)
    return starts


def get_patch_locs(
    p_shape: tuple[int, ...], p_ovl: tuple[int, ...], v_shape: tuple[int, ...]
) -> NDArray[np.int32]:
    """
    Get all the patch top-left corner locations.

    Parameters
    ----------
    vol_shape : tuple
        The volume shape
    patch_shape : tuple
        The patch shape
    patch_overlap : tuple
        The overlap of patch for each dimension.

    Returns
    -------
    numpy.ndarray
        All the patch top-left corner locations.

    Notes
    -----
    This is a legacy function, you probably want to use the PatchedArray class.
    """
    # Create an iterator for all the possible patches top-left corner location.
    if len(v_shape) != len(p_shape) or len(v_shape) != len(p_ovl):
        raise ValueError(
            f"Dimension mismatch between the arguments. {p_shape}{p_ovl}, {v_shape}"
        )

    ranges = []
    for v_s, p_s, p_o in zip(v_shape, p_shape, p_ovl):
        if p_o >= p_s:
            raise ValueError(
                "Overlap should be a non-negative integer smaller than patch_size",
            )
        ranges.append(patch_starts(v_s, p_s, p_s - p_o))
    # fast ND-Cartesian product from https://stackoverflow.com/a/11146645
    patch_locs = np.empty(
        [len(arr) for arr in ranges] + [len(p_shape)],
        dtype=np.int32,
    )
    for idx, coords in enumerate(np.ix_(*ranges)):
        patch_locs[..., idx] = coords

    return patch_locs.reshape(-1, len(p_shape))


def svd_analysis(
    input_data: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return the SVD decomposition of the patch, centered along time.

    U, S, Vt and M are compute such that:
    ``X = U @ (S * Vt) + M``, with ``M`` the temporal mean of each voxel.

    Removing the temporal mean makes the constant time course an exact null
    direction of the centered patch: its rank is at most ``min(N, T - 1)``, and
    only these ``k = min(N, T - 1)`` components are returned. The noise of the
    centered patch is then an i.i.d. ``N x (T - 1)`` matrix, so downstream
    thresholds use ``k`` as the time dimension.

    Parameters
    ----------
    input_data : numpy.ndarray
        The patch, of shape ``(N, T)`` (voxels x time).

    Returns
    -------
    u_vec, s_vals, v_vec, mean
        ``(N, k)``, ``(k,)``, ``(k, T)`` and ``(N, 1)`` arrays.
    """
    mean = np.mean(input_data, axis=1, keepdims=True)
    data_centered = input_data - mean
    u_vec, s_vals, v_vec = svd(data_centered, full_matrices=False)
    k = n_components(input_data.shape)
    return u_vec[:, :k], s_vals[:k], v_vec[:k], mean


def n_components(patch_shape: tuple[int, ...]) -> int:
    """Return the number of components of a ``(N, T)`` patch centered along time."""
    return min(patch_shape[0], patch_shape[1] - 1)


def svd_synthesis(
    u_vec: NDArray,
    s_vals: NDArray,
    v_vec: NDArray,
    mean: NDArray,
    idx: int,
) -> NDArray:
    """
    Reconstruct ``X = (U @ (S * V)) + M`` with only the max_idx greatest component.

    U, S, V must be sorted in decreasing order.

    Parameters
    ----------
    u_vec : numpy.ndarray
    s_vals : numpy.ndarray
    v_vec : numpy.ndarray
    mean : numpy.ndarray
    idx : int

    Returns
    -------
    np.ndarray: The reconstructed matrix.
    """
    return (u_vec[:, :idx] @ (s_vals[:idx, None] * v_vec[:idx, :])) + mean


def eig_analysis(
    input_data: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Return the eigen values and vectors of the autocorrelation of the patch.

    This method is faster than the svd, but the eigen values
    are in increasing order. As in :func:`svd_analysis`, the patch is centered
    along time and the null direction this creates is excluded: at most
    ``min(N, T - 1)`` eigenvalues are returned.

    Parameters
    ----------
    input_data : np.ndarray
        A 2D Array, of shape ``(N, T)`` (voxels x time).

    Returns
    -------
    A : numpy.ndarray
        The centered patch A = X - M
    d : numpy.ndarray
        The eigenvalues of A^H A / N
    W : numpy.ndarray
        The eigenvector matrix of A^H A
    M : numpy.ndarray
        The temporal mean of each voxel, of shape ``(N, 1)``.
    """
    mean = np.mean(input_data, axis=1, keepdims=True)
    data_centered = input_data - mean
    N, T = data_centered.shape
    n_eig = n_components(input_data.shape)
    eig_vals, eig_vec = eigh(
        data_centered.conj().T @ data_centered / N,
        driver="evx",
        subset_by_index=(T - n_eig, T - 1),
    )

    return data_centered, eig_vals, eig_vec, mean


def eig_synthesis(
    data_centered: NDArray,
    eig_vec: NDArray,
    mean: NDArray,
    max_val: int,
) -> NDArray:
    """Reconstruction the denoise patch with truncated eigen decomposition.

    This implements equations (1) and (2) of manjon2013
    """
    eig_vec[:, :-max_val] = 0
    return ((data_centered @ eig_vec) @ eig_vec.conj().T) + mean


def marchenko_pastur_median(beta: float, eps: float = 1e-7) -> float:
    r"""Compute the median of the Marchenko-Pastur Distribution.

    Parameters
    ----------
    beta: float
        aspect ratio of a matrix.
    eps: float
        Precision Parameter
    Return
    ------
    float: the estimated median

    Notes
    -----
    This method Solve :math:`F(x) = 1/2` by dichotomy with
    .. math ::

        F(x) = \int_{\beta_-}^{x} \frac{\sqrt{(\beta_+-t)(t-\beta_-)}}{2\pi\beta t} dt

    The integral is computed using scipy.integrate.quad
    """
    if not (0 <= beta <= 1):
        raise ValueError("Aspect Ratio should be between 0 and 1")

    beta_p = (1 + np.sqrt(beta)) ** 2
    beta_m = (1 - np.sqrt(beta)) ** 2

    def mp_pdf(x: float) -> float:
        """Marchenko Pastur Probability density function."""
        if beta_p >= x >= beta_m:
            return np.sqrt((beta_p - x) * (x - beta_m)) / (2 * np.pi * x * beta)
        else:
            return 0

    change = True
    hibnd = beta_p
    lobnd = beta_m
    # quad return (value, upperbound_error).
    # We only need the integral value

    n = 0
    while change and (hibnd - lobnd) > eps and n < 20:
        change = False
        midpoints = np.linspace(lobnd, hibnd, 5)
        int_estimates = np.array(
            list(map(lambda xx: quad(lambda x: mp_pdf(x), beta_m, xx)[0], midpoints))
        )
        if np.any(int_estimates < 0.5):
            lobnd = np.max(midpoints[int_estimates < 0.5])
            change = True
        if np.any(int_estimates > 0.5):
            hibnd = np.min(midpoints[int_estimates > 0.5])
            change = True
        n += 1
    return (lobnd + hibnd) / 2


def estimate_noise(noise_sequence: NDArray, block_size: int = 1) -> NDArray:
    """Estimate a noise map from a noise only sequence.

    The noise map is the standard deviation of the noise in each patch.

    Parameters
    ----------
    noise_sequence : np.ndarray of shape (X, Y, Z, T)
        The noise-only data.
    block_size : int
        The size of the patch used to estimate the noise.

    Returns
    -------
    np.ndarray of shape (X, Y, Z)
        The estimated noise map.
    """
    volume_shape = noise_sequence.shape[:-1]
    noise_map = np.empty(volume_shape)
    patch_shape = (block_size,) * len(volume_shape)
    patch_overlap = (block_size - 1,) * len(volume_shape)

    for patch_tl in get_patch_locs(patch_shape, patch_overlap, volume_shape):
        # Get the index of voxels in the patch
        patch_slice = tuple(
            slice(ptl, ptl + ps) for ptl, ps in zip(patch_tl, patch_shape)
        )
        # Identify the voxel in the center of the patch
        patch_center_img = tuple(
            slice(ptl + ps // 2, ptl + ps // 2 + 1)
            for ptl, ps in zip(patch_tl, patch_shape)
        )
        # Set the value of the voxel in the center of the patch to the SD of
        # the patch
        noise_map[patch_center_img] = np.std(noise_sequence[patch_slice])
    return noise_map
