"""Torch module for denoising a batch of patches using local low rank method."""

import numpy as np
import torch

from ..space_time.utils import marchenko_pastur_median


def _center_indices(patch_shape: tuple[int, ...]) -> tuple[int, int]:
    """Flattened spatial index and time index of a patch's center voxel.

    Used by "center" recombination: only that single voxel of each denoised
    patch is ever kept, so only the row of ``u`` and column of ``v`` that
    produce it need to be computed.
    """
    spatial_shape = patch_shape[:-1]
    spatial_idx = 0
    for c, s in zip((p // 2 for p in spatial_shape), spatial_shape):
        spatial_idx = spatial_idx * s + c
    return spatial_idx, patch_shape[-1] // 2


class OptimalSVDDenoiser(torch.nn.Module):
    """Optimal SVD denoiser for a batch of patches (Optimized for torch.compile)."""

    def __init__(
        self,
        patch_shape,
        recombination="weighted",
        loss="fro",
        eps_marshenko_pastur=1e-7,
        full_time=False,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.recombination = recombination
        self.loss = loss
        # True when the patch spans the full data extent on the time axis
        # (e.g. patch_shape[-1] == -1 resolved to the full length): "center"
        # recombination then keeps the whole time profile at the spatial
        # center instead of collapsing it to a single time point too.
        self.full_time = full_time
        self.center_spatial_idx, self.center_time_idx = _center_indices(patch_shape)

        if loss not in ["fro", "nuc", "ope"]:
            raise ValueError(f"Invalid loss {loss}, must be 'fro', 'nuc', or 'ope'")

        beta = patch_shape[-1] / np.prod(patch_shape[:-1])

        # Precompute all constants to save math ops in the forward pass. .
        self.beta = float(beta)
        self.sqrt_beta = float(np.sqrt(beta))
        self.mp_median = float(
            np.sqrt(marchenko_pastur_median(beta=beta, eps=eps_marshenko_pastur))
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

    def forward(self, x: torch.Tensor, var_apriori: torch.Tensor | None = None):
        """Apply optimal SVD denoising to a batch of patches.

        Parameters
        ----------
        x : (B, *patch_shape) tensor
            Batch of patches to denoise.
        var_apriori : (B,) tensor, optional
            Per-patch noise variance (mean of the squared noise std over the
            patch footprint), matching CPU's ``noise_std`` path. If None,
            sigma is self-estimated from the median singular value
            (Marchenko-Pastur), matching CPU's ``noise_std=None`` path.

        Returns
        -------
        x_denoised : (B, *patch_shape) tensor
            Denoised patches.
        weight : (B,) tensor
            Per-patch recombination weight, for weighted patch recombination.
        var_estimate : (B,) tensor
            Per-patch noise variance estimate.
        maxidx : (B,) tensor
            Per-patch rank after denoising.
        """
        # Flatten and mean center
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, N, T)
        n_dim = x_flat.shape[-2]
        m = torch.mean(x_flat, dim=-2, keepdim=True)
        xc = x_flat - m
        u, s, v = torch.linalg.svd(xc, full_matrices=False, driver="gesvda")

        if var_apriori is not None:
            sigma = torch.sqrt(var_apriori)
        else:
            n_svals = s.shape[-1]
            # manual median because s is already sorted.
            lo, hi = (n_svals - 1) // 2, n_svals // 2
            median_s = 0.5 * (s[..., lo] + s[..., hi])
            sigma = median_s / (self.mp_median * (n_dim**0.5))
        scale_factor = sigma * (n_dim**0.5)

        # Apply shrink
        scale_factor_exp = scale_factor.unsqueeze(-1)
        s_shrink = self._shrink(s / scale_factor_exp)
        s_shrink = s_shrink * scale_factor_exp
        s_shrink = torch.nan_to_num(s_shrink, nan=0.0)

        maxidx = torch.sum(s_shrink > 0, dim=-1)

        if self.recombination == "center":
            # Only the spatial center of the reconstructed patch is ever kept
            # (see base.py's Recombination.CENTER), so skip the full
            # (B, N, T) reconstruction and project just the row of `u` that
            # produces it.
            u_center = u[:, self.center_spatial_idx, :]
            if self.full_time:
                # Time axis spans the whole data extent: keep the full
                # reconstructed time profile at the spatial center instead
                # of also collapsing it to a single time point.
                x_center = (
                    torch.matmul((u_center * s_shrink).unsqueeze(1), v).squeeze(1)
                    + m[:, 0, :]
                )
                return x_center, 1, sigma**2, maxidx.to(torch.int32)
            v_center = v[:, :, self.center_time_idx]
            x_center = (
                torch.sum(u_center * s_shrink * v_center, dim=-1)
                + m[:, 0, self.center_time_idx]
            )
            return x_center, 1, sigma**2, maxidx.to(torch.int32)

        if self.recombination == "weighted":
            weight = 1.0 / (2.0 + maxidx)
        else:
            weight = torch.ones_like(maxidx, dtype=torch.float32)

        x_denoised = torch.matmul(u * s_shrink.unsqueeze(1), v) + m

        return (
            x_denoised.reshape(x.shape),
            weight,
            sigma**2,
            maxidx.to(torch.int32),
        )


class MPPCADenoiser(torch.nn.Module):
    """MP PCA denoiser."""

    def __init__(
        self,
        patch_shape,
        recombination="weighted",
        threshold_scale=1.0,
        full_time=False,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.threshold_scale = threshold_scale
        self.recombination = recombination
        self.full_time = full_time
        self.center_spatial_idx, self.center_time_idx = _center_indices(patch_shape)

    def forward(self, x: torch.Tensor):
        """Apply MP PCA denoising to a batch of patches."""
        # Flatten and mean center
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, N,M)

        xm = torch.mean(x_flat, dim=-2, keepdim=True)
        xc = x_flat - xm

        u, s, v = torch.linalg.svd(xc, full_matrices=False, driver="gesvda")

        N, M = x_flat.shape[-2], x_flat.shape[-1]
        # Convert singular values to eigenvalues of covariance
        eigs = s**2 / (N - 1)
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

        p_range = torch.arange(M, device=x.device)
        # eigs is ascending, so mask is True for all indices < p, and False after
        mask = ((eigs - eigs[:, -1:]) * (M - p_range) * (N - p_range)) > (
            4 * rcum_eigs * (M * N) ** 0.5 * self.threshold_scale**2
        )
        p = torch.sum(mask, dim=-1)  # p is the index of the last True in mask
        eigs = eigs * (p_range < p.unsqueeze(-1))
        s_shrink = torch.sqrt(eigs * (N - 1))

        batch_idx = torch.arange(x_flat.shape[0], device=x.device)
        var_estimate = rcum_eigs[batch_idx, p] / (M - p)

        if self.recombination == "center":
            u_center = u[:, self.center_spatial_idx, :]
            if self.full_time:
                x_center = (
                    torch.matmul((u_center * s_shrink).unsqueeze(1), v).squeeze(1)
                    + xm[:, 0, :]
                )
                return x_center, 1, var_estimate, p
            v_center = v[:, :, self.center_time_idx]
            x_center = (
                torch.sum(u_center * s_shrink * v_center, dim=-1)
                + xm[:, 0, self.center_time_idx]
            )
            return x_center, 1, var_estimate, p

        x_denoised = torch.matmul(u * s_shrink.unsqueeze(1), v) + xm

        if self.recombination == "weighted":
            weight = 1.0 / (2.0 + p)
        else:
            weight = torch.ones_like(p, dtype=torch.float32)

        return x_denoised.reshape(x.shape), weight, var_estimate, p.to(torch.int32)
