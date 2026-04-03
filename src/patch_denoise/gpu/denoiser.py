"""Torch module for denoising a batch of patches using local low rank method."""

import logging
import torch
import numpy as np

from ..space_time.utils import marchenko_pastur_median


class OptimalSVDDenoiser(torch.nn.Module):
    """Optimal SVD denoiser for a batch of patches (Optimized for torch.compile)."""

    def __init__(
        self,
        patch_shape,
        recombination="weighted",
        loss="fro",
        eps_marshenko_pastur=1e-7,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.recombination = recombination
        self.loss = loss

        if loss not in ["fro", "nuc", "ope"]:
            raise ValueError(f"Invalid loss {loss}, must be 'fro', 'nuc', or 'ope'")

        beta = patch_shape[-1] / np.prod(patch_shape[:-1])
        mp_median_val = np.sqrt(
            marchenko_pastur_median(beta=beta, eps=eps_marshenko_pastur)
        )

        # Precompute all constants to save math ops in the forward pass
        self.beta = torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.mp_median = torch.nn.Parameter(
            torch.tensor(mp_median_val, dtype=torch.float32)
        )

    def _opt_loss_x(self, y):
        """Compute (8) of donoho2017 using precomputed buffers."""
        tmp = y**2 - self.beta - 1.0
        # Use boolean to float conversion instead of boolean indexing
        mask = (y >= (1 + torch.sqrt(self.beta))).to(y.dtype)
        return torch.sqrt(0.5 * (tmp + torch.sqrt((tmp**2) - 4 * self.beta))) * mask

    def _shrink(self, singvals):
        """Apply the selected shrinkage function."""
        if self.loss == "ope":
            return torch.nn.functional.relu(self._opt_loss_x(singvals))

        elif self.loss == "nuc":
            tmp = self._opt_loss_x(singvals)
            return torch.nn.functional.relu(
                tmp**4 - (torch.sqrt(self.beta) * tmp * singvals) - self.beta
            ) / ((tmp**2) * singvals)

        elif self.loss == "fro":
            return torch.sqrt(
                torch.nn.functional.relu(
                    ((singvals**2) - self.beta - 1.0) ** 2 - 4 * self.beta
                )
                / singvals
            )

    def forward(self, x: torch.Tensor):
        """Apply optimal SVD denoising to a batch of patches."""
        # Flatten and mean center
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, N, T)
        m = torch.mean(x_flat, dim=-2, keepdim=True)
        xc = x_flat - m
        u, s, v = torch.linalg.svd(xc, full_matrices=False, driver="gesvda")
        # Calculate noise scale
        median_s = torch.median(s, dim=-1)[0]
        scale_factor = median_s / self.mp_median  # type: ignore

        # 3. Use unsqueeze for clarity and predictable dimension expansion
        scale_factor_exp = scale_factor.unsqueeze(-1)

        # Apply shrink
        s_shrink = self._shrink(s / scale_factor_exp)
        s_shrink = s_shrink * scale_factor_exp

        # 4. Remove boolean indexing to prevent graph breaks. Use nan_to_num.
        s_shrink = torch.nan_to_num(s_shrink, nan=0.0)

        # 5. Simplify rank logic and weighting
        rank = torch.sum(s_shrink > 0, dim=-1) + 1

        if self.recombination == "weighted":
            weight = 1.0 / rank
        else:
            weight = torch.ones_like(rank, dtype=x.dtype)

        # unsqueeze does broadcasting, no copies.
        x_denoised = torch.matmul(u * s_shrink.unsqueeze(1), v) + m

        return x_denoised.reshape(x.shape), weight


class MPPCADenoiser(torch.nn.Module):
    """MP PCA denoiser."""

    def __init__(
        self,
        patch_shape,
        recombination="weighted",
        threshold_scale=1.0,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.threshold_scale = threshold_scale
        self.recombination = recombination

    def forward(self, x: torch.Tensor):
        """Apply MP PCA denoising to a batch of patches."""
        # Flatten and mean center
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, N,M)

        xm = torch.mean(x_flat, dim=-1, keepdim=True)
        xc = x_flat - xm

        u, s, v = torch.linalg.svd(xc, full_matrices=False, driver="gesvda")

        N, M = x_flat.shape[-2], x_flat.shape[-1]
        # Convert singular values to eigenvalues of covariance
        eigs = s**2 / (N - 1)
        # NB: The singular values are returned in descending order.
        # create a reverse order cum sum
        cum_eigs = torch.cumsum(eigs, dim=-1)
        rcum_eigs = eigs - cum_eigs + cum_eigs[:, -1:]

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

        p = torch.arange(M)
        p = torch.argmax(
            (eigs - eigs[:, -1]) * (M - p) * (N - p)
            < 4 * rcum_eigs * torch.sqrt(M * N),  # type: ignore
            dim=-1,
        )
        eigs[:, p:] = 0
        s_shrink = torch.sqrt(eigs * (N - 1))
        x_denoised = torch.matmul(u * s_shrink.unsqueeze(1), v) + xm

        if self.recombination == "weighted":
            weight = 1.0 / (1.0 + p)
        else:
            weight = torch.ones_like(p, dtype=x.dtype)

        return x_denoised.reshape(x.shape), weight
