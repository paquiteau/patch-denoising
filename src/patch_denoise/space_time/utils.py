"""Utilities for space-time denoising."""
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh, svd
import cupy as cp


def svd_analysis(input_data):
    """Return the centered SVD decomposition.

    U, S, Vt and M are compute such that:
    ``X = U @ (S * Vt) + M.``

    Parameters
    ----------
    input_data : numpy.ndarray
        The patch

    Returns
    -------
    u_vec, s_vals, v_vec, mean
    """
    # TODO  benchmark svd vs svds and order of data.
    mean = np.mean(input_data, axis=0)
    data_centered = input_data - mean
    u_vec, s_vals, v_vec = svd(data_centered, full_matrices=False)

    return u_vec, s_vals, v_vec, mean


def svd_analysis_gpu(input_data, batch_size):
    total_samples = input_data.shape[0]
    num_batches = int(np.ceil(total_samples/ batch_size))
    adjusted_batch_size = total_samples // num_batches
    last_batch_size = total_samples % adjusted_batch_size

    # Initialize arrays to store the results
    # input_data shape is (total patches, patch size, time)
    m = input_data.shape[1]
    n = input_data.shape[2]
    U_batched = cp.empty((total_samples, m, n), dtype=cp.float64)
    S_batched = cp.empty((total_samples, min(m, n)), dtype=cp.float64)
    V_batched = cp.empty((total_samples, n, n), dtype=cp.float64)
    mean_batched = cp.empty((total_samples, n), dtype=cp.float64)

    # Compute SVD for each matrix in the batch
    for i in range(num_batches):
        print(i)
        start_idx = i * adjusted_batch_size
        end_idx = start_idx + adjusted_batch_size if i < num_batches - 1 else start_idx + last_batch_size
        idx = slice(start_idx, end_idx)
        mean = cp.mean(input_data[idx], axis=1, keepdims=True)
        data_centered = cp.asarray(input_data[idx] - mean)
        u_vec, s_vals, v_vec = cp.linalg.svd(
            data_centered, full_matrices=False
        )
        U_batched[idx] = u_vec
        S_batched[idx] = s_vals
        V_batched[idx] = v_vec
        mean_batched[idx] = cp.asarray(cp.squeeze(mean))
    return U_batched, S_batched, V_batched, mean_batched


def svd_synthesis(u_vec, s_vals, v_vec, mean, idx):
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


def eig_analysis(input_data, max_eig_val=10):
    """
    Return the eigen values and vectors of the autocorrelation of the patch.

    This method is faster than the svd, but the eigen values
    are in increasing order.

    Parameters
    ----------
    input_data : np.ndarray
        A 2D Array
    max_eig_val : int, optional
       For faster results, only the ``max_eig_val`` biggest eigenvalues are
       computed. default = 10

    Returns
    -------
    A : numpy.ndarray
        The centered patch A = X - M
    d : numpy.ndarray
        The eigenvalues of A^H A
    W : numpy.ndarray
        The eigenvector matrix of A^H A
    M : numpy.ndarray
        The mean of the patch along the time axis
    """
    mean = np.mean(input_data, axis=0)
    data_centered = input_data - mean
    M, N = data_centered.shape
    eig_vals, eig_vec = eigh(
        data_centered.conj().T @ data_centered / (M - 1),
        driver="evx",
        subset_by_index=(len(mean) - max_eig_val, len(mean) - 1),
    )

    return data_centered, eig_vals, eig_vec, mean


def eig_synthesis(data_centered, eig_vec, mean, max_val):
    """Reconstruction the denoise patch with truncated eigen decomposition.

    This implements equations (1) and (2) of manjon2013
    """
    eig_vec[:, :-max_val] = 0
    return ((data_centered @ eig_vec) @ eig_vec.conj().T) + mean


def marshenko_pastur_median(beta, eps=1e-7):
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

    def mp_pdf(x):
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


def get_patch_locs(p_shape, p_ovl, v_shape):
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
    """
    # Create an iterator for all the possible patches top-left corner location.
    if len(v_shape) != len(p_shape) or len(v_shape) != len(p_ovl):
        raise ValueError("Dimension mismatch between the arguments.")

    ranges = []
    for v_s, p_s, p_o in zip(v_shape, p_shape, p_ovl):
        if p_o >= p_s:
            raise ValueError(
                "Overlap should be a non-negative integer smaller than patch_size",
            )
        last_idx = v_s - p_s
        range_ = np.arange(0, last_idx, p_s - p_o, dtype=np.int32)
        if range_[-1] < last_idx:
            range_ = np.append(range_, last_idx)
        ranges.append(range_)
    # fast ND-Cartesian product from https://stackoverflow.com/a/11146645
    patch_locs = np.empty(
        [len(arr) for arr in ranges] + [len(p_shape)],
        dtype=np.int32,
    )
    for idx, coords in enumerate(np.ix_(*ranges)):
        patch_locs[..., idx] = coords

    return patch_locs.reshape(-1, len(p_shape))


def get_patches_gpu(input_data, patch_shape, patch_overlap):
    """Extract all the patches from a volume.
    
    Returns
    -------
    numpy.ndarray
        All the patches in shape (patches, patch size, time).
    """
    patch_size = np.prod(patch_shape)

    # Pad the data
    input_data = cp.asarray(input_data)

    c, h, w, t_s = input_data.shape
    kc, kh, kw = patch_shape  # kernel size
    sc, sh, sw = np.repeat(
        patch_shape[0] - patch_overlap[0], len(patch_shape)
    )
    needed_c = int((cp.ceil((c - kc) / sc + 1) - ((c - kc) / sc + 1)) * kc)
    needed_h = int((cp.ceil((h - kh) / sh + 1) - ((h - kh) / sh + 1)) * kh)
    needed_w = int((cp.ceil((w - kw) / sw + 1) - ((w - kw) / sw + 1)) * kw)

    input_data_padded = cp.pad(
        input_data, ((0, needed_c), (0, needed_h), (0, needed_w), (0, 0)
    ), mode='edge')

    step = patch_shape[0] - patch_overlap[0]
    patches = cp.lib.stride_tricks.sliding_window_view(
        input_data_padded, patch_shape, axis=(0, 1, 2)
    )[::step, ::step, ::step]

    patches = patches.transpose((0, 1, 2, 4, 5, 6, 3))
    patches = patches.reshape((np.prod(patches.shape[:3]), patch_size, t_s))
    
    return cp.asnumpy(patches)


def estimate_noise(noise_sequence, block_size=1):
    """Estimate the temporal noise standard deviation of a noise only sequence."""
    volume_shape = noise_sequence.shape[:-1]
    noise_map = np.empty(volume_shape)
    patch_shape = (block_size,) * len(volume_shape)
    patch_overlap = (block_size - 1,) * len(volume_shape)

    for patch_tl in get_patch_locs(patch_shape, patch_overlap, volume_shape):
        patch_slice = tuple(
            slice(ptl, ptl + ps) for ptl, ps in zip(patch_tl, patch_shape)
        )
        patch_center_img = tuple(
            slice(ptl + ps // 2, ptl + ps // 2 + 1)
            for ptl, ps in zip(patch_tl, patch_shape)
        )
        noise_map[patch_center_img] = np.std(noise_sequence[patch_slice])
    return noise_map
