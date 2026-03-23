"""Torch modules responsible for performing the denoising of patches."""

import torch
import numpy as np

from ..space_time.utils import marchenko_pastur_median


def _opt_loss_x(y, beta):
    """Compute (8) of donoho2017."""
    tmp = y**2 - beta - 1
    return torch.sqrt(0.5 * (tmp + torch.sqrt((tmp**2) - (4 * beta)))) * (
        y >= (1 + torch.sqrt(beta))
    )


def _opt_ope_shrink(singvals, beta=1):
    """Perform batched optimal threshold of singular values for operator norm."""
    return torch.maximum(_opt_loss_x(singvals, beta), 0, dim=-1)


def _opt_nuc_shrink(singvals, beta=1):
    """Perform batched optimal threshold of singular values for nuclear norm."""
    tmp = _opt_loss_x(singvals, beta)
    return torch.maximum(
        0,
        (tmp**4 - (torch.sqrt(beta) * tmp * singvals) - beta),
        dim=-1,
    ) / ((tmp**2) * singvals)


def _opt_fro_shrink(singvals, beta=1):
    """Perform batched optimal threshold of singular values for frobenius norm."""
    return torch.sqrt(
        torch.maximum(
            (((singvals**2) - beta - 1) ** 2 - 4 * beta),
            0,
            dim=-1,
        )
        / singvals
    )


_OPT_LOSS_SHRINK = {
    "fro": _opt_fro_shrink,
    "nuc": _opt_nuc_shrink,
    "ope": _opt_ope_shrink,
}


class OptimalSVDDenoiser(torch.nn.Module):
    """Optimal SVD denoiser for a batch of patches."""

    def __init__(
        self,
        patch_shape,
        recombination="weighted",
        loss="fro",
        eps_marshenko_pastur=1e-7,
    ):
        super().__init__()
        self.patch_shape = patch_shape

        if loss not in _OPT_LOSS_SHRINK:
            raise ValueError(
                f"Invalid loss {loss}, must be one of {list(_OPT_LOSS_SHRINK.keys())}"
            )
        self.shrink_func = _OPT_LOSS_SHRINK[loss]

        self.mp_median = torch.Tensor(
            marchenko_pastur_median(
                beta=patch_shape[-1] / patch_shape, eps=eps_marshenko_pastur
            )
        )

        # TODO: add apriori variance utilization

    def forward(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
    ):
        """Denoise a batch of patches using optimal SVD shrinkage."""
        B, *XYZ, T = x.shape
        N = torch.prod(XYZ)
        x_flat = x.reshape(B, -1, T)
        m = torch.mean(x_flat, dim=-1, keepdim=True)
        xc = x_flat - m
        u, s, v = torch.linalg.svd(xc, full_matrices=False)

        # sigma = torch.median(s, dim=-1)/torch.sqrt(T * self.mp_median)
        # scale_factor = torch.sqrt(T)*sigma
        scale_factor = torch.median(s, dim=-1) / torch.sqrt(self.mp_median)

        s_shrink = self.shrink_func(s / scale_factor, beta=N / T) * scale_factor
        s_shrink *= scale_factor

        s_shrink[torch.isnan(s_shrink)] = 0  # Handle NaNs from zero division

        rank = torch.sum(s_shrink > 0, dim=-1) + 1

        x_denoised = (u * s_shrink) @ v + m

        if self.recombination == "weighted":
            # Weigh the denoised patch by the number of retained singular values
            weight = 1 / (2 + rank)
            x_denoised *= weight
            return x_denoised, weight
        elif self.recombination == "average":
            return x_denoised, torch.ones(B, device=x.device)
        else:
            raise ValueError(f"Unknown recombination method {self.recombination}")
