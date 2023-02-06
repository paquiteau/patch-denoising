"""Low Rank Denoising methods."""
from types import MappingProxyType

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

from .base import BaseSpaceTimeDenoiser
from .utils import (
    eig_analysis,
    eig_synthesis,
    marshenko_pastur_median,
    svd_analysis,
    svd_synthesis,
)
from .._docs import fill_doc

NUMBA_AVAILABLE = True
try:
    import numba as nb
except ImportError:
    NUMBA_AVAILABLE = False
    pass


@fill_doc
class MPPCADenoiser(BaseSpaceTimeDenoiser):
    """Denoising using the MP-PCA threshoding.

    Parameters
    ----------
    $patch_config
    threshold_scale: float
        An extra factor multiplying the threshold.
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


@fill_doc
class HybridPCADenoiser(BaseSpaceTimeDenoiser):
    """Denoising using the Hybrid-PCA thresholding method.

    Parameters
    ----------
    $patch_config
    """

    @fill_doc
    def denoise(
        self, input_data, mask=None, mask_threshold=50, noise_std=1.0, progbar=None
    ):
        """Denoise using the Hybrid-PCA method.

        Along with the input data a noise std map or value should be provided.

        Parameters
        ----------
        $input_config
        $mask_config
        $noise_std

        Returns
        -------
        $denoise_return
        """
        if isinstance(noise_std, (float, np.floating)):
            self.input_denoising_kwargs["var_apriori"] = noise_std**2 * np.ones(
                input_data.shape[:-1]
            )
        else:
            self.input_denoising_kwargs["var_apriori"] = noise_std**2

        return super().denoise(input_data, mask, mask_threshold, progbar=progbar)

    def _patch_processing(self, patch, patch_slice=None, var_apriori=None):
        """Process a pach with the Hybrid-PCA method."""
        varest = np.mean(var_apriori[patch_slice])
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


@fill_doc
class RawSVDDenoiser(BaseSpaceTimeDenoiser):
    """
    Classical Patch wise singular value thresholding denoiser.

    Parameters
    ----------
    $patch_config
    threshold_vlue: float
        treshold value for the singular values.
    """

    def __init__(
        self, patch_shape, patch_overlap, threshold_value=1.0, recombination="weighted"
    ):
        self._threshold_val = threshold_value

        super().__init__(patch_shape, patch_overlap, recombination)

    @fill_doc
    def denoise(
        self,
        input_data,
        mask=None,
        mask_threshold=50,
        threshold_scale=1.0,
        progbar=None,
    ):
        """Denoise the input_data, according to mask.

        Patches are extracted sequentially and process by the implemented
        `_patch_processing` function.
        Only patches which have at least a voxel in the mask ROI are processed.

        Parameters
        ----------
        $input_config
        $mask_config
        threshold_scale: float
            Extra factor for the threshold of singular values.

        Returns
        -------
        $denoise_return
        """
        self._threshold = self._threshold_val * threshold_scale
        return super().denoise(input_data, mask, mask_threshold, progbar=progbar)

    def _patch_processing(self, patch, patch_slice=None, **kwargs):
        """Process a pach with the simple SVT method."""
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


@fill_doc
class NordicDenoiser(RawSVDDenoiser):
    """Denoising using the NORDIC method.

    Parameters
    ----------
    $patch_config
    """

    @fill_doc
    def denoise(
        self,
        input_data,
        mask=None,
        mask_threshold=50,
        noise_std=1.0,
        n_iter_threshold=10,
        progbar=None,
    ):
        """Denoise using the NORDIC method.

        Along with the input data a noise stp map or value should be provided.

        Parameters
        ----------
        $input_config
        $mask_config
        $noise_std

        Returns
        -------
        $denoise_return

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
            input_data, mask, mask_threshold=mask_threshold, progbar=progbar
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


@fill_doc
class OptimalSVDDenoiser(BaseSpaceTimeDenoiser):
    """
    Optimal Shrinkage of singular values for a specific norm.

    Parameters
    ----------
    $patch_config
    loss: str
        The loss determines the choise of the optimal thresholding function
        associated to it. The losses `"fro"`, `"nuc"` and `"op"` are supported,
        for the frobenius, nuclear and operator norm, respectively.
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
        super().__init__(patch_shape, patch_overlap, recombination=recombination)
        self.input_denoising_kwargs[
            "shrink_func"
        ] = OptimalSVDDenoiser._OPT_LOSS_SHRINK[loss]

    @fill_doc
    def denoise(
        self,
        input_data,
        mask=None,
        mask_threshold=50,
        noise_std=None,
        eps_marshenko_pastur=1e-7,
        progbar=None,
    ):
        """
        Optimal thresholing denoising method.

        Parameters
        ----------
        $input_config
        $mask_config
        $noise_std
        loss: str
            The loss for which the optimal thresholding is perform.
        eps_marshenko_pastur: float
            The precision with which the optimal threshold is computed.

        Returns
        -------
        $denoise_return

        Notes
        -----
        Reimplement of the original Matlab code [#]_ in python.

        References
        ----------
        .. [#] Gavish, Matan, and David L. Donoho. \
            "Optimal Shrinkage of Singular Values."
            IEEE Transactions on Information Theory 63, no. 4 (April 2017): 2137–52.
            https://doi.org/10.1109/TIT.2017.2653801.
        """
        patch_shape, _ = self._BaseSpaceTimeDenoiser__get_patch_param(input_data.shape)

        self.input_denoising_kwargs["mp_median"] = marshenko_pastur_median(
            beta=input_data.shape[-1] / np.prod(patch_shape),
            eps=eps_marshenko_pastur,
        )
        if noise_std is None:
            self.input_denoising_kwargs["var_apriori"] = None
        elif isinstance(noise_std, (float, np.floating)):
            self.input_denoising_kwargs["var_apriori"] = noise_std**2 * np.ones(
                input_data.shape[:-1]
            )
        else:
            self.input_denoising_kwargs["var_apriori"] = noise_std**2

        return super().denoise(input_data, mask, mask_threshold, progbar=progbar)

    def _patch_processing(
        self,
        patch,
        patch_slice=None,
        shrink_func=None,
        mp_median=None,
        var_apriori=None,
    ):
        u_vec, s_values, v_vec, p_tmean = svd_analysis(patch)
        if var_apriori is not None:
            sigma = np.mean(np.sqrt(var_apriori[patch_slice]))
        else:
            sigma = np.median(s_values) / np.sqrt(patch.shape[1] * mp_median)

        scale_factor = np.sqrt(patch.shape[1]) * sigma
        thresh_s_values = scale_factor * shrink_func(
            s_values / scale_factor,
            beta=patch.shape[1] / patch.shape[0],
        )
        thresh_s_values[np.isnan(thresh_s_values)] = 0

        if np.any(thresh_s_values):
            maxidx = np.max(np.nonzero(thresh_s_values)) + 1
            p_new = svd_synthesis(u_vec, thresh_s_values, v_vec, p_tmean, maxidx)
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
    n, p = np.shape(X)
    if method == "qut":
        gamma = np.exp(gamma) + 1
    else:
        tau = np.exp(tau)

    sing_vals2 = sing_vals**2
    n_vals = len(sing_vals)
    D = np.zeros((n_vals, n_vals), dtype=np.float32)
    dhat = sing_vals * np.maximum(1 - ((tau / sing_vals) ** gamma), 0)
    tmp = sing_vals * dhat
    for i in range(n_vals):
        diff2i = sing_vals2[i] - sing_vals2
        diff2i[i] = np.inf
        D[i, :] = tmp[i] / diff2i

    gradd = (1 + (gamma - 1) * (tau / sing_vals) ** gamma) * (sing_vals >= tau)
    div = np.sum(gradd + abs(n - p) * dhat / sing_vals) + 2 * np.sum(D)

    rss = np.sum((dhat - sing_vals) ** 2)
    if method == "gsure":
        return rss / (1 - div / n / p) ** 2
    return (sigma**2) * ((-n * p) + (2 * div)) + rss


if NUMBA_AVAILABLE:
    s = nb.float32
    d = nb.float64
    sure_atn_cost = nb.njit(
        [
            s(s[:, :], nb.types.unicode_type, s[:], s, s, s),
            s(s[:, :], nb.types.unicode_type, s[:], s, d, d),
        ],
        fastmath=True,
    )(_sure_atn_cost)


def _atn_shrink(singvals, gamma, tau):
    """Adaptive trace norm shrinkage."""
    return singvals * np.maximum(1 - (tau / singvals) ** gamma, 0)


def _get_gamma_tau_qut(patch, sing_vals, stdest, gamma0, nbsim):
    """Estimate gamma and tau using the quantile method."""
    maxd = np.ones(nbsim)
    for i in range(nbsim):
        maxd[i] = np.max(
            svd(
                np.random.randn(*patch.shape) * stdest,
                compute_uv=False,
                overwrite_a=True,
            )
        )
    # auto estimation of tau.
    tau = np.quantile(maxd, 1 - 1 / np.sqrt(np.log(max(*patch.shape))))
    # single value for gamma not provided, estimating it.
    if not isinstance(gamma0, (float, np.floating)):

        def sure_gamma(gamma):
            return _sure_atn_cost(
                X=patch,
                method="qut",
                sing_vals=sing_vals,
                gamma=gamma,
                sigma=stdest,
                tau=tau,
            )

        res_opti = minimize(sure_gamma, 0)
        gamma = np.exp(res_opti.x) + 1
    else:
        gamma = gamma0
    return gamma, tau


def _get_gamma_tau(patch, sing_vals, stdest, method, gamma0, tau0):
    """Estimate gamma and tau."""
    # estimation of tau
    def sure_tau(tau, *args):
        return _sure_atn_cost(*args, tau[0])

    if tau0 is None:
        tau0 = np.log(np.median(sing_vals))
    cost_glob = np.Inf
    for g in gamma0:
        res_opti = minimize(
            lambda x: _sure_atn_cost(
                X=patch,
                method=method,
                gamma=g,  # noqa: B023
                sing_vals=sing_vals,
                sigma=stdest,
                tau=x,
            ),
            tau0,
        )
        # get cost value.
        cost = _sure_atn_cost(
            X=patch,
            method=method,
            gamma=g,
            sing_vals=sing_vals,
            sigma=stdest,
            tau=res_opti.x,
        )
        if cost < cost_glob:
            gamma = g
            tau = np.exp(res_opti.x)
            cost_glob = cost
    return gamma, tau


@fill_doc
class AdaptiveDenoiser(BaseSpaceTimeDenoiser):
    """Adaptive Denoiser.

    Parameters
    ----------
    $patch_config
    """

    _SUPPORTED_METHOD = ["sure", "qut", "gsure"]

    def __init__(
        self,
        patch_shape,
        patch_overlap,
        method="SURE",
        recombination="weighted",
        nbsim=500,
    ):
        super().__init__(patch_shape, patch_overlap, recombination)
        if method.lower() not in self._SUPPORTED_METHOD:
            raise ValueError(
                f"Unsupported method: '{method}', use any of {self._SUPPORTED_METHOD}"
            )
        self.input_denoising_kwargs["method"] = method.lower()
        self.input_denoising_kwargs["nbsim"] = nbsim

    @fill_doc
    def denoise(
        self,
        input_data,
        mask=None,
        mask_threshold=50,
        tau0=None,
        noise_std=None,
        gamma0=None,
        progbar=None,
    ):
        """
        Adaptive denoiser.

        Perform the denoising using the adaptive trace norm estimator. [#]_

        Parameters
        ----------
        $input_config
        $mask_config
        $noise_std


        References
        ----------
        .. [#] J. Josse and S. Sardy, “Adaptive Shrinkage of singular values.”
            arXiv, Nov. 22, 2014.
            doi: 10.48550/arXiv.1310.6602.


        """
        self.input_denoising_kwargs["gamma0"] = gamma0
        self.input_denoising_kwargs["tau0"] = tau0

        if isinstance(noise_std, (float, np.floating)):
            self.input_denoising_kwargs["var_apriori"] = noise_std**2 * np.ones(
                input_data.shape[:-1]
            )
        else:
            self.input_denoising_kwargs["var_apriori"] = noise_std**2
        return super().denoise(input_data, mask, mask_threshold, progbar=progbar)

    def _patch_processing(
        self,
        patch,
        patch_slice=None,
        gamma0=None,
        tau0=None,
        var_apriori=None,
        method=None,
        nbsim=None,
    ):
        stdest = np.sqrt(np.mean(var_apriori[patch_slice]))

        u_vec, sing_vals, v_vec, p_tmean = svd_analysis(patch)

        if method == "qut":
            gamma, tau = _get_gamma_tau_qut(patch, sing_vals, stdest, gamma0, nbsim)
        else:
            gamma, tau = _get_gamma_tau(patch, sing_vals, stdest, method, gamma0, tau0)
        # end of parameter selection
        # Perform thresholding

        thresh_s_values = _atn_shrink(sing_vals, gamma=gamma, tau=tau)
        if np.any(thresh_s_values):
            maxidx = np.max(np.nonzero(thresh_s_values)) + 1
            p_new = svd_synthesis(u_vec, thresh_s_values, v_vec, p_tmean, maxidx)
        else:
            maxidx = 0
            p_new = np.zeros_like(patch) + p_tmean

        # Equation (3) in Manjon 2013
        theta = 1.0 / (1.0 + maxidx)
        p_new *= theta
        weights = theta

        return p_new, weights, np.NaN
