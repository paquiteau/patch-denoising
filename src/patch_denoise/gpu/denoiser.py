"""Torch module for denoising a batch of patches using local low rank method."""

import numpy as np
import torch

from ..space_time.utils import marchenko_pastur_median
from ._svd import FastPatchSVD


def _center_indices(patch_shape: tuple[int, ...]) -> tuple[int, int]:
    """Flattened spatial index and time index of a patch's center voxel.

    Used by "center" recombination: only that single voxel of each denoised
    patch is ever kept, so only its row/column need to be computed -- see
    ``FastPatchSVD.center_reconstruct`` in ``_svd.py``.
    """
    spatial_shape = patch_shape[:-1]
    spatial_idx = 0
    for c, s in zip((p // 2 for p in spatial_shape), spatial_shape):
        spatial_idx = spatial_idx * s + c
    return spatial_idx, patch_shape[-1] // 2


class _LowRankDenoiser(torch.nn.Module):
    """Shared driver: eigendecompose, shrink the spectrum, reconstruct.

    Subclasses only implement :meth:`_shrink_spectrum`.
    """

    def __init__(self, patch_shape, recombination="weighted", full_time=False):
        super().__init__()
        self.patch_shape = patch_shape
        self.recombination = recombination
        # True when the patch spans the full data extent on the time axis
        # (e.g. patch_shape[-1] == -1 resolved to the full length): "center"
        # recombination then keeps the whole time profile at the spatial
        # center instead of collapsing it to a single time point too.
        self.center_spatial_idx, self.center_time_idx = _center_indices(patch_shape)
        if full_time:
            self.center_time_idx = None
        self.N: int = int(np.prod(patch_shape[:-1]))
        self.T: int = patch_shape[-1]
        self._svd = FastPatchSVD(self.T)

    def _shrink_spectrum(
        self, s: torch.Tensor, var_apriori: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the shrinkage ratio ``s_shrink / s``, the noise variance and rank.

        ``s`` is the ``(B, k)`` descending spectrum of the patches centered
        along time.
        """
        raise NotImplementedError

    def _weight(self, rank: torch.Tensor) -> torch.Tensor:
        if self.recombination == "weighted":
            return 1.0 / (2.0 + rank)
        return torch.ones_like(rank, dtype=torch.float32)

    def forward(self, x: torch.Tensor, var_apriori: torch.Tensor | None = None):
        """Denoise a batch of patches.

        Parameters
        ----------
        x : ``(B, *patch_shape)`` tensor
            Batch of patches to denoise.
        var_apriori : (B,) tensor, optional
            Per-patch noise variance, if the method uses it.

        Returns
        -------
        x_denoised : ``(B, *patch_shape)`` tensor
            Denoised patches; for "center" recombination, only the center
            voxel (or time profile, with ``full_time``) of each patch.
        weight : (B,) tensor
            Per-patch recombination weight.
        var_estimate : (B,) tensor
            Per-patch noise variance estimate.
        rank : (B,) int32 tensor
            Per-patch rank after denoising.
        """
        x_flat = x.reshape(x.shape[0], self.N, self.T)
        s, vh, m, xc = self._svd.eigh(x_flat)
        ratio, var, rank = self._shrink_spectrum(s, var_apriori)

        if self.recombination == "center":
            x_center = self._svd.center_reconstruct(
                x_flat, vh, ratio, self.center_spatial_idx, self.center_time_idx
            )
            return x_center, 1, var, rank

        x_denoised = self._svd.reconstruct(x_flat, m, xc, vh, ratio)
        return x_denoised.reshape(x.shape), self._weight(rank), var, rank

    def forward_patches(
        self,
        vol: torch.Tensor,
        bases: torch.Tensor,
        out: torch.Tensor,
        wsum: torch.Tensor,
        var_apriori: torch.Tensor | None = None,
    ):
        """Denoise patches read in place from ``vol``, accumulated in place.

        Never materializes the batch of patches nor of their reconstruction:
        see ``FastPatchSVD.eigh_patches``/``reconstruct_patches``. Weighted/
        average recombination only.

        Parameters
        ----------
        vol : (X, Y, Z, T') tensor, float32 or complex64
            Volume the patches are read from, shifted by (about) the temporal
            mean of each voxel for accuracy, see ``_svd.py``.
        bases : (B,) int64 tensor
            Element offset in ``vol`` of the first voxel of each patch.
        out : tensor, same dtype as ``vol``
        wsum : tensor, float32
            Accumulators for the weighted reconstruction and the weights,
            with the same shape and strides as ``vol``.
        var_apriori : (B,) tensor, optional
            As for :meth:`forward`.

        Returns
        -------
        weight, var_estimate, rank :
            As for :meth:`forward`.
        """
        s, vh = self._svd.eigh_patches(vol, bases, self.patch_shape)
        ratio, var, rank = self._shrink_spectrum(s, var_apriori)
        weight = self._weight(rank)
        self._svd.reconstruct_patches(
            vol, bases, self.patch_shape, vh, ratio, weight, out, wsum
        )
        return weight, var, rank

    def forward_center(
        self,
        vol: torch.Tensor,
        bases: torch.Tensor,
        var_apriori: torch.Tensor | None = None,
    ):
        """Denoise the center of patches read in place from ``vol``.

        "center" recombination counterpart of :meth:`forward_patches`: only
        the time profile at the spatial center of each patch is reconstructed,
        as a ``(B, T)`` tensor; which of its frames are kept is up to the
        caller (see ``center_time_slice``). Otherwise returns as :meth:`forward`.
        """
        s, vh = self._svd.eigh_patches(vol, bases, self.patch_shape)
        ratio, var, rank = self._shrink_spectrum(s, var_apriori)
        center = sum(
            (p // 2) * st for p, st in zip(self.patch_shape[:-1], vol.stride()[:-1])
        )
        row_idx = (bases + center)[:, None] + vol.stride(-1) * torch.arange(
            self.T, device=vol.device
        )
        x_row = vol.as_strided((vol.numel(),), (1,))[row_idx]
        x_center = self._svd.reconstruct_row(x_row, vh, ratio)
        return x_center, 1, var, rank


class OptimalSVDDenoiser(_LowRankDenoiser):
    """Optimal SVD denoiser for a batch of patches."""

    def __init__(
        self,
        patch_shape,
        recombination="weighted",
        loss="fro",
        eps_marshenko_pastur=1e-7,
        full_time=False,
    ):
        super().__init__(patch_shape, recombination, full_time)
        self.loss = loss

        if loss not in ["fro", "nuc", "ope"]:
            raise ValueError(f"Invalid loss {loss}, must be 'fro', 'nuc', or 'ope'")

        # Patches are centered along time: the noise is an N x (T - 1) matrix.
        self.M: int = self.T - 1

        self.beta = float(self.M / self.N)

        # Precompute all constants to save math ops in the forward pass. .
        self.sqrt_beta = float(np.sqrt(self.beta))
        self.sqrt_mp_med = float(
            np.sqrt(marchenko_pastur_median(beta=self.beta, eps=eps_marshenko_pastur))
        )
        self.mp_edge = 1.0 + self.sqrt_beta  # upper Marchenko-Pastur edge
        self.mp_edge_hi = self.mp_edge**2  # squared edges, for the "fro" branch
        self.mp_edge_lo = (1.0 - self.sqrt_beta) ** 2

    def _opt_loss_x(self, y):
        """Compute (8) of donoho2017 using precomputed buffers."""
        tmp = y**2 - self.beta - 1.0
        # Use boolean to float conversion instead of boolean indexing
        mask = (y >= self.mp_edge).to(y.dtype)
        return torch.sqrt(0.5 * (tmp + torch.sqrt((tmp**2) - 4 * self.beta))) * mask

    def _shrink(self, singvals):
        """Apply the selected shrinkage function."""
        if self.loss == "ope":
            return torch.nn.functional.relu(self._opt_loss_x(singvals))

        elif self.loss == "nuc":
            tmp = self._opt_loss_x(singvals)
            return torch.nn.functional.relu(
                tmp**4 - (self.sqrt_beta * tmp * singvals) - self.beta
            ) / ((tmp**2) * singvals)

        elif self.loss == "fro":
            # eta(y) = sqrt((y**2 - beta - 1)**2 - 4 beta) / y
            # for y >= 1 + sqrt(beta), 0 otherwise.
            # Factorized as (y**2 - hi)(y**2 - lo)/y with hi/lo the squared
            # MP edges. relu on (y**2 - hi) enforces constraint.
            y2 = singvals * singvals
            above = torch.nn.functional.relu(y2 - self.mp_edge_hi)
            return torch.sqrt(above * (y2 - self.mp_edge_lo)) / singvals.clamp_min(
                torch.finfo(singvals.dtype).tiny
            )

    def _shrink_spectrum(self, s, var_apriori):
        """Optimal shrinkage (Gavish & Donoho), see :meth:`forward`.

        ``var_apriori`` is the per-patch noise variance (mean of the squared
        noise std over the patch footprint), matching CPU's ``noise_std``
        path. If None, sigma is self-estimated from the median singular value
        (Marchenko-Pastur), matching CPU's ``noise_std=None`` path.
        """
        if var_apriori is not None:
            sigma = torch.sqrt(var_apriori)
            scale_factor = sigma * (self.M**0.5)
        else:
            # manual median because s is already sorted.
            k = s.shape[-1]
            lo, hi = (k - 1) // 2, k // 2
            # compute the estimator y_med / sqrt(med_mp), and the associated
            # scale factor to apply to the singular values before shrinkage.
            scale_factor = s[..., lo] + s[..., hi]
            scale_factor /= 2 * self.sqrt_mp_med
            sigma = scale_factor / (self.M**0.5)

        # Apply shrink
        scale_factor_exp = scale_factor.unsqueeze(-1)
        s_shrink = self._shrink(s / scale_factor_exp)
        s_shrink = s_shrink * scale_factor_exp
        s_shrink = torch.nan_to_num(s_shrink, nan=0.0)

        maxidx = torch.sum(s_shrink > 0, dim=-1)
        s_safe = s.clamp_min(torch.finfo(s.dtype).tiny)
        ratio = (s_shrink / s_safe).to(s.dtype, copy=False)
        return ratio, sigma**2, maxidx.to(torch.int32)


class MPPCADenoiser(_LowRankDenoiser):
    """MP PCA denoiser."""

    def __init__(
        self,
        patch_shape,
        recombination="weighted",
        threshold_scale=1.0,
        full_time=False,
    ):
        super().__init__(patch_shape, recombination, full_time)
        self.threshold_scale = threshold_scale

    def _shrink_spectrum(self, s, var_apriori=None):
        """MP-PCA hard thresholding (Veraart et al. 2016); ignores ``var_apriori``."""
        # Centering along time leaves an N x (T - 1) noise matrix: M = T - 1.
        N, M = self.N, s.shape[-1]
        # Convert singular values to eigenvalues of covariance
        eigs = s**2 / N
        # NB: The singular values are returned in descending order.
        # create a reverse order cum sum
        cum_eigs = torch.cumsum(eigs, dim=-1)
        rcum_eigs = eigs - cum_eigs + cum_eigs[:, -1:]

        # Original Matlab code for reference:
        # [lambda,order] = sort(lambda,'descend');
        # U = U(:,order);
        # csum = cumsum(lambda,'reverse');
        # p = (0:length(lambda)-1)';
        # p = -1 + find((lambda-lambda(end)).*(M-p).*(N-p) < 4*csum*sqrt(M*N),1);
        # if p==0
        #     X = zeros(size(X));
        # elseif M<N
        #     X = U(:,1:p)*U(:,1:p)'*X;
        # else
        #     X = X*U(:,1:p)*U(:,1:p)';
        # end
        # s2 = csum(p+1)/((M-p)*(N-p));
        # s2_after = s2 - csum(p+1)/(M*N);

        p_range = torch.arange(M, device=s.device)
        # eigs is ascending, so mask is True for all indices < p, and False after
        mask = ((eigs - eigs[:, -1:]) * (M - p_range) * (N - p_range)) > (
            4 * rcum_eigs * (M * N) ** 0.5 * self.threshold_scale**2
        )
        p = torch.sum(mask, dim=-1)  # p is the index of the last True in mask
        eigs = eigs * (p_range < p.unsqueeze(-1))
        s_shrink = torch.sqrt(eigs * (N - 1))

        batch_idx = torch.arange(s.shape[0], device=s.device)
        var_estimate = rcum_eigs[batch_idx, p] / (M - p)

        s_safe = s.clamp_min(torch.finfo(s.dtype).tiny)
        ratio = s_shrink / s_safe
        return ratio, var_estimate, p.to(torch.int32)
