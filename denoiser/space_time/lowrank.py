"""Low Rank Denoising methods."""
import numpy as np
from .base import BaseSpaceTimeDenoiser
from .utils import (
    eig_analysis,
    eig_synthesis,
    marshenko_pastur_median,
    svd_analysis,
    svd_synthesis,
)
from scipy.linalg import svd

from types import MappingProxyType


class MPPCADenoiser(BaseSpaceTimeDenoiser):
    """Denoising using the MP-PCA threshoding.

    Parameters
    ----------
    patch_shape : tuple
        The patch shape
    patch_overlap : tuple
        The amount of overlap between patches in each direction
    recombination : str
        The method of reweighting patches. either "weighed" or "average"
    """

    def __init__(self, patch_shape, patch_overlap, threshold_scale, **kwargs):
        super().__init__(patch_shape, patch_overlap, **kwargs)
        self.input_denoising_kwargs["threshold_scale"] = threshold_scale

    def _patch_processing(self, patch, patch_slice=None, threshold_scale=1.0):
        """Process a pach with the MP-PCA method."""
        p_center, eig_vals, eig_vec, p_tmean = eig_analysis(patch)
        maxidx = 0
        meanvar = np.mean(eig_vals)
        meanvar *= 4 * np.sqrt((len(eig_vals) - maxidx + 1) / len(patch))
        while meanvar < eig_vals[~maxidx] - eig_vals[0]:
            maxidx += 1
            meanvar = np.mean(eig_vals[:-maxidx])
            meanvar *= 4 * np.sqrt((len(eig_vec) - maxidx + 1) / len(patch))
        var_noise = np.mean(eig_vals[: len(eig_vals) - maxidx])

        maxidx = np.sum(eig_vals > (var_noise * threshold_scale**2))

        if maxidx == 0:
            patch_new = np.zeros_like(patch) + p_tmean
        else:
            patch_new = eig_synthesis(p_center, eig_vec, p_tmean, maxidx)

        # Equation (3) of Manjon 2013
        weights = 1.0 / (1.0 + maxidx)
        noise_map = var_noise * weights
        patch_new *= weights

        return patch_new, noise_map, weights


class HybridPCADenoiser(BaseSpaceTimeDenoiser):
    """Denoising using the Hybrid-PCA thresholding method.

    Parameters
    ----------
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        The amount of overlap between patches in each direction
    recombination: str
        The method of reweighting patches. either "weighed" or "average"
    """

    def denoise(self, input_data, mask=None, mask_threshold=50, noise_std=1.0):
        """Denoise using the Hybrid-PCA method.

        Along with the input data a noise std map or value should be provided.
        """
        if isinstance(noise_std, (float, np.floating)):
            self._noise_apriori = noise_std**2 * np.ones(input_data.shape[:-1])
        else:
            self._noise_apriori = noise_std**2

        return super().denoise(input_data, mask, mask_threshold)

    def _patch_processing(self, patch, patch_slice=None):
        """Process a pach with the Hybrid-PCA method."""
        varest = np.mean(self._noise_apriori[patch_slice])
        p_center, eig_vals, eig_vec, p_tmean = eig_analysis(patch)
        maxidx = 0
        var_noise = np.mean(eig_vals)
        while var_noise > varest and maxidx < len(eig_vals) - 2:
            maxidx += 1
            var_noise = np.mean(eig_vals[:-maxidx])
        if maxidx == 0:  # all eigen values are noise
            patch_new = np.zeros_like(patch) + p_tmean
        else:
            patch_new = eig_synthesis(p_center, eig_vec, p_tmean, maxidx)
        # Equation (3) of Manjon2013
        weights = 1.0 / (1.0 + maxidx)
        noise_map = var_noise * weights
        patch_new *= weights

        return patch_new, noise_map, weights


class RawSVDDenoiser(BaseSpaceTimeDenoiser):
    """
    Classical Patch wise singular value thresholding denoiser.

    Parameters
    ----------
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        The amount of overlap between patches in each direction
    recombination: str
        The method of reweighting patches. either "weighed" or "average"
    """

    def __init__(
        self, patch_shape, patch_overlap, threshold_value=1.0, recombination="weighted"
    ):
        self._threshold_val = threshold_value

        super().__init__(patch_shape, patch_overlap, recombination)

    def denoise(self, input_data, mask=None, mask_threshold=50, threshold_scale=1.0):
        self._threshold = self._threshold_val * threshold_scale
        return super().denoiser(input_data, mask, mask_threshold)

    def _patch_processing(self, patch, patch_slice=None, **kwargs):
        """
        Denoise a patch using the singular value thresholding.

        Parameters
        ----------
        patch : numpy.ndarray
            The patch to process
        threshold_value : float
            The thresholding value for the patch

        Returns
        -------
        patch_new : numpy.ndarray
            The processed patch.
        weights : numpy.ndarray
            The weight associated with the patch.
        """
        # Centering for better precision in SVD
        u_vec, s_values, v_vec, p_tmean = svd_analysis(patch)

        maxidx = np.sum(s_values > self._threshold)
        if maxidx == 0:
            p_new = np.zeros_like(patch) + p_tmean
        else:
            s_values[s_values < self._threshold] = 0
            p_new = svd_synthesis(u_vec, s_values, v_vec, p_tmean, maxidx)

        # Equation (3) in Manjon 2013
        theta = 1.0 / (1.0 + maxidx)
        p_new *= theta
        weights = theta

        return p_new, weights, np.NaN


class NordicDenoiser(RawSVDDenoiser):
    """Denoising using the Hybrid-PCA thresholding method.

    Parameters
    ----------
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        The amount of overlap between patches in each direction
    recombination: str
        The method of reweighting patches. either "weighed" or "average"
    """

    def denoise(
        self,
        input_data,
        mask=None,
        mask_threshold=50,
        noise_std=1.0,
        n_iter_threshold=10,
    ):
        """Denoise using the NORDIC method.

        Along with the input data a noise stp map or value should be provided.
        """
        patch_shape, _ = self._BaseSpaceTimeDenoiser__get_patch_param(input_data.shape)
        # compute the threshold using Monte-Carlo Simulations.
        max_sval = sum(
            max(
                svd(
                    np.random.randn(np.prod(patch_shape), input_data.shape[-1]),
                    compute_uv=False,
                )
            )
            for _ in range(n_iter_threshold)
        )
        max_sval /= n_iter_threshold

        if isinstance(noise_std, np.ndarray):
            noise_std = np.mean(noise_std)
        if not isinstance(noise_std, (float, np.floating)):
            raise ValueError(
                "For NORDIC the noise level must be either an"
                + " array or a float specifying the std in the volume.",
            )

        self._threshold = noise_std * max_sval

        return super(RawSVDDenoiser, self).denoise(
            input_data, mask, mask_threshold=mask_threshold
        )


# From MATLAB implementation
def _opt_loss_x(y, beta):
    """Compute (8) of donoho2017."""
    tmp = y**2 - beta - 1
    return np.sqrt(0.5 * (tmp + np.sqrt((tmp**2) - (4 * beta)))) * (
        y >= (1 + np.sqrt(beta))
    )


def _opt_ope_shrink(singvals, beta=1):
    """Perform optimal threshold of singular values for operator norm."""
    return np.maximum(_opt_loss_x(singvals, beta), 0)


def _opt_nuc_shrink(singvals, beta=1):
    """Perform optimal threshold of singular values for nuclear norm."""
    tmp = _opt_loss_x(singvals, beta)
    return np.maximum(
        0,
        (tmp**4 - (np.sqrt(beta) * tmp * singvals) - beta),
    ) / ((tmp**2) * singvals)


def _opt_fro_shrink(singvals, beta=1):
    """Perform optimal threshold of singular values for frobenius norm."""
    return np.sqrt(
        np.maximum(
            (((singvals**2) - beta - 1) ** 2 - 4 * beta),
            0,
        )
        / singvals
    )


class OptimalSVDDenoiser(BaseSpaceTimeDenoiser):
    """
    Optimal Shrinkage of singular values for a specific norm.

    Parameters
    ----------
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        The amount of overlap between patches in each direction
    loss: str
        The loss determines the choise of the optimal thresholding function
        associated to it. The losses `"fro"`, `"nuc"` and `"op"` are supported,
        for the frobenius, nuclear and operator norm, respectively.
    recombination: str
        The method of reweighting patches. either "weighted" or "average"
    """

    _OPT_LOSS_SHRINK = MappingProxyType(
        {
            "fro": _opt_fro_shrink,
            "nuc": _opt_nuc_shrink,
            "ope": _opt_ope_shrink,
        }
    )

    def __init__(
        self,
        patch_shape,
        patch_overlap,
        loss="fro",
        recombination="weighted",
    ):

        self._opt_loss_shrink = OptimalSVDDenoiser._OPT_LOSS_SHRINK[loss]

        super().__init__(patch_shape, patch_overlap, recombination=recombination)

    def denoise(
        self,
        input_data,
        mask=None,
        mask_threshold=50,
        eps_marshenko_pastur=1e-7,
    ):

        patch_shape, _ = self._BaseSpaceTimeDenoiser__get_patch_param(input_data.shape)
        self._mp_median = marshenko_pastur_median(
            beta=input_data.shape[-1] / np.prod(patch_shape),
            eps=eps_marshenko_pastur,
        )

        return super().denoise(input_data, mask, mask_threshold)

    def _patch_processing(self, patch, patch_slice=None, **kwargs):

        u_vec, s_values, v_vec, p_tmean = svd_analysis(patch)

        sigma = np.median(s_values) / self._mp_median

        thresh_s_values = sigma * self._opt_loss_shrink(s_values / sigma)

        if np.any(thresh_s_values):
            maxidx = np.max(np.nonzero(thresh_s_values)) + 1
            p_new = svd_synthesis(u_vec, s_values, v_vec, p_tmean, maxidx)
        else:
            maxidx = 0
            p_new = np.zeros_like(patch) + p_tmean

        # Equation (3) in Manjon 2013
        theta = 1.0 / (1.0 + maxidx)
        p_new *= theta
        weights = theta

        return p_new, weights, np.NaN


def _sure_atn_cost(X, method, sing_vals, gamma, sigma=None, tau=None):
    """
    Compute the SURE cost function.

    Parameters
    ----------
    X: np.ndarray
    sing_vals : singular values of X
    gamma: float
    sigma: float
    tau: float
    """


def _adaptive_trace_norm_shrink(singvals, gamma, tau):
    """Adaptive trace shrinkage."""
    return singvals * np.maximum(1 - (tau / singvals) ** gamma, 0)


class AdaptiveDenoiser(BaseSpaceTimeDenoiser):
    """Adaptive Denoiser.

    Parameters
    ----------
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        The amount of overlap between patches in each direction
    recombination: str
        The method of reweighting patches. either "weighted" or "average"
    """

    def __init__(
        self,
        patch_shape,
        patch_overlap,
        method="SURE",
        recombination="weighted",
    ):
        if method.lower() == "sure":
            ...
        elif method.lower() == "gsure":
            ...
        elif method.lower() == "qut":
            ...
        else:
            raise ValueError(f"Unsupported method '{method}'")

    def denoise(
        self,
        input_data,
        mask=None,
        mask_threshold=50,
    ):

        patch_shape, _ = self._BaseSpaceTimeDenoiser__get_patch_param(input_data.shape)

        return super().denoise(input_data, mask, mask_threshold)
