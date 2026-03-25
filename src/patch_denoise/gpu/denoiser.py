"""Torch module for denoising a batch of patches using local low rank method."""

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

        # 1. Register buffers so they live on the GPU and move with module.cuda()
        # Precompute all constants to save math ops in the forward pass
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer(
            "mp_median", torch.tensor(mp_median_val, dtype=torch.float32)
        )
        # Precompute constants for the optimal shrinkage functions to avoid redundant calculations in the forward pass
        self.register_buffer("sqrt_beta", torch.sqrt(self.beta))
        self.register_buffer("four_beta", 4.0 * self.beta)
        self.register_buffer("threshold", 1.0 + self.sqrt_beta)

    def _opt_loss_x(self, y):
        """Compute (8) of donoho2017 using precomputed buffers."""
        tmp = y**2 - self.beta - 1.0
        # Use boolean to float conversion instead of boolean indexing
        mask = (y >= self.threshold).to(y.dtype)
        return torch.sqrt(0.5 * (tmp + torch.sqrt((tmp**2) - self.four_beta))) * mask

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
            return torch.sqrt(
                torch.nn.functional.relu(
                    (((singvals**2) - self.beta - 1.0) ** 2 - self.four_beta)
                )
                / singvals
            )

    def forward(self, x: torch.Tensor):
        # Flatten and mean center
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, N, T)
        m = torch.mean(x_flat, dim=-1, keepdim=True)
        xc = x_flat - m

        u, s, v = torch.linalg.svd(xc, full_matrices=False)

        # Calculate noise scale
        median_s = torch.median(s, dim=-1)[0]
        scale_factor = median_s / self.mp_median

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
            weight = 1.0 / (2.0 + rank)
        else:
            weight = torch.ones_like(rank, dtype=x.dtype)

        # 6. Reconstruct using matmul (more compile-friendly than @ operator in some edge cases)
        # u * s_shrink.unsqueeze(1) relies on broadcasting, avoiding large intermediate allocations
        x_denoised = torch.matmul(u * s_shrink.unsqueeze(1), v) + m

        x_denoised = x_denoised * weight.unsqueeze(-1).unsqueeze(-1)

        return x_denoised, weight
