import numpy as np
from scipy.linalg import svd, eigh
from scipy.integrate import quad


def svd_analysis(input_data):
    """Return the centered SVD decomposition  X = U @ (S * Vt) + M.

    Parameters
    ----------
    input_data : numpy.ndarray
        The patch

    Returns
    -------
    u_vec, s_vals, v_vec, mean
    """
    mean = np.mean(input_data, axis=0)
    input_data -= mean
    # TODO  benchmark svd vs svds and order of data.
    u_vec, s_vals, v_vec = svd(input_data, full_matrices=False)

    return u_vec, s_vals, v_vec, mean


def svd_synthesis(u_vec, s_vals, v_vec, mean, idx):
    """
    Reconstruct X= (U @ (S * V)) + M with only the max_idx greatest component.

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

    This method is surprisingly faster than the svd, but the eigen values
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
    eig_vals, eig_vec = eigh(
        data_centered.conj().T @ data_centered,
        turbo=True,
        subset_by_index=(len(mean) - max_eig_val, len(mean) - 1),
    )

    return data_centered, eig_vals, eig_vec, mean


def eig_synthesis(data_centered, eig_vec, mean, max_val):
    """Reconstruction the denoise patch with truncated eigen decomposition.

    This implements equations (1) and (2) of :cite:`manjon2013`
    """
    eig_vec[:, :-max_val] = 0
    return ((data_centered @ eig_vec) @ eig_vec.conj().T) + mean


def marshenko_pastur_median(beta, eps=1e-7):
    r"""Compute the median of the Marchenko-Pastur Distribution.

    Parameters
    ---------
    beta: float
        aspect ratio of a matrix.
    eps: float
        Precision Parameter
    Return
    ------
    float: the estimated median

    Notes
    -----
    This method Solve F(x) = 1/2 by dichotomy with
    .. math ::

    F(x) = \int_{\beta_{-}}^{x} \frac{\sqrt{(\beta_{+}-t)(t-\beta_{-})}}{2\pi\beta t} \mathrm{d}t

    The integral is computed using scipy.integrate.quad
    """
    if not (0 <= beta <= 1):
        raise ValueError("Aspect Ratio should be between 0 and 1")

    beta_p = (1 + np.sqrt(beta)) ** 2
    beta_m = (1 - np.sqrt(beta)) ** 2

    def mp_pdf(x):
        """Marchenko Pastur Probability density function"""
        if beta_p >= x >= beta_m:
            return np.sqrt((beta_p - x) * (x - beta_m)) / (2 * np.pi * x * beta)
        else:
            return 0

    change = True
    hibnd = beta_p
    lobnd = beta_m
    # quad return (value, upperbound_error).
    # We only need the integral value
    func = lambda xx: quad(lambda x: mp_pdf(x), beta_m, xx)[0]

    n = 0
    while change and (hibnd - lobnd) > eps and n < 20:
        change = False
        midpoints = np.linspace(lobnd, hibnd, 5)
        int_estimates = np.array(list(map(func, midpoints)))
        if np.any(int_estimates < 0.5):
            lobnd = np.max(midpoints[int_estimates < 0.5])
            change = True
        if np.any(int_estimates > 0.5):
            hibnd = np.min(midpoints[int_estimates > 0.5])
            change = True
        n += 1
    return (lobnd + hibnd) / 2