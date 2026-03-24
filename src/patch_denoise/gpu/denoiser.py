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
    return torch.nn.functional.relu(_opt_loss_x(singvals, beta))


def _opt_nuc_shrink(singvals, beta=1):
    """Perform batched optimal threshold of singular values for nuclear norm."""
    tmp = _opt_loss_x(singvals, beta)
    return torch.nn.functional.relu(
        tmp**4 - (torch.sqrt(beta) * tmp * singvals) - beta
    ) / ((tmp**2) * singvals)


def _opt_fro_shrink(singvals, beta=1):
    """Perform batched optimal threshold of singular values for frobenius norm."""
    return torch.sqrt(
        torch.nn.functional.relu(
            (((singvals**2) - beta - 1) ** 2 - 4 * beta),
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
        batch_size,
        recombination="weighted",
        loss="fro",
        eps_marshenko_pastur=1e-7,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.recombination = recombination
        if loss not in _OPT_LOSS_SHRINK:
            raise ValueError(
                f"Invalid loss {loss}, must be one of {list(_OPT_LOSS_SHRINK.keys())}"
            )
        self.shrink_func = _OPT_LOSS_SHRINK[loss]

        beta = patch_shape[-1] / np.prod(patch_shape[:-1])
        self.mp_median = torch.tensor(
            np.sqrt(
                marchenko_pastur_median(
                    beta=beta,
                    eps=eps_marshenko_pastur,
                )
            )
        )
        self.beta = torch.tensor(beta)
        # TODO: add apriori variance utilization
        self.weight_fun = lambda rank: (
            1 / (2 + rank)
            if recombination == "weighted"
            else lambda rank: torch.ones(batch_size, device=rank.device)
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Denoise a batch of patches using optimal SVD shrinkage."""
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, N, T)
        m = torch.mean(x_flat, dim=-1, keepdim=True)
        xc = x_flat - m
        u, s, v = torch.linalg.svd(xc, full_matrices=False)

        # sigma = torch.median(s, dim=-1)/torch.sqrt(T * self.mp_median)
        # scale_factor = torch.sqrt(T)*sigma
        scale_factor = torch.median(s, dim=-1)[0] / self.mp_median

        s_shrink = self.shrink_func(s / scale_factor[:, None], beta=self.beta)
        s_shrink *= scale_factor[:, None]

        s_shrink[torch.isnan(s_shrink)] = 0  # Handle NaNs from zero division

        rank = torch.sum(s_shrink > 0, dim=-1) + 1

        x_denoised = (u * s_shrink[:, None, :]) @ v + m

        # Weigh the denoised patch by the number of retained singular values
        weight = self.weight_fun(rank)  # 1 / (2 + rank)
        x_denoised *= weight[:, None, None]
        return x_denoised, weight
