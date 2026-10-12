"""Torch dataloader for the noisy data."""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ..space_time.utils import patch_starts


def patch_grid(data_shape, patch_shape, patch_overlap, device=None):
    """Start index of every patch, along each axis.

    See :func:`~patch_denoise.space_time.utils.patch_starts`: the last patch of
    each axis is aligned on the border, so that every voxel is covered.

    Returns
    -------
    list of (n_d,) int64 tensors
        The patch grid is their Cartesian product, in C order.
    """
    step = np.subtract(patch_shape, patch_overlap)
    if np.any(step < 0):
        raise ValueError("overlap should be smaller than patch on every dimension.")
    if not len(data_shape) == len(patch_shape) == len(patch_overlap):
        raise ValueError(
            "patch_shape and patch_overlap must have the same number of dimensions "
            "as the input data."
        )
    return [
        torch.from_numpy(patch_starts(ds, ps, st)).to(device)
        for ds, ps, st in zip(data_shape, patch_shape, step)
    ]


def sliding_sum_nd(data, patch_shape, starts):
    """
    Sum ``data`` over every patch of the grid, with separable 1D convolutions.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor.
    patch_shape : tuple of int
        The shape of each patch.
    starts : list of (n_d,) int64 tensors
        Start index of the patches along each axis, see :func:`patch_grid`.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_0, n_1, ...)``: the sum of ``data`` over each patch
        of the grid.

    Notes
    -----
    The problem is separable: along each axis, the sum over every window (a
    stride-1 convolution) is computed, then only the patch starts are kept.
    This is (significantly) more memory efficient than gathering patches.
    """
    res = data
    for dim in range(res.ndim):
        k = patch_shape[dim]
        if k > 1:
            # Move the target dimension last, the other ones are seen as batch.
            moved = res.movedim(dim, -1)
            weight = torch.ones((1, 1, k), dtype=res.dtype, device=res.device)
            windows = F.conv1d(moved.reshape(-1, 1, moved.shape[-1]), weight)
            res = windows.view(*moved.shape[:-1], -1).movedim(-1, dim)
        res = res.index_select(dim, starts[dim])
    return res


def select_patches_to_process(mask, patch_shape, starts, mask_threshold=50):
    """Select patches to process based on the mask and threshold.

    Parameters
    ----------
    mask : torch.Tensor
        The input mask, with the same shape as the input data.
    patch_shape : tuple of int
        The shape of each patch.
    starts : list of (n_d,) int64 tensors
        Start index of the patches along each axis, see :func:`patch_grid`.
    mask_threshold : float, default 50
        The percentage threshold for selecting patches. Patches with a
        percentage of masked pixels above this threshold will be selected for
        processing.

    Returns
    -------
    torch.Tensor
        Linear indices, in the patch grid, of the patches to process.
    """
    with torch.inference_mode():
        patch_score = sliding_sum_nd(mask.to(torch.float32), patch_shape, starts)
        patch_score /= np.prod(patch_shape)
        return torch.where(patch_score.ravel() > mask_threshold / 100)[0]


class PatchDataset:
    """GPU-resident collection of patches, batch-gathered from ``input_data``.

    ``input_data`` is expected to already live on the target device: patches
    are then extracted with a single vectorized advanced-index gather per
    batch (see :meth:`get_batch`), with no host-device transfer or per-item
    Python loop in the hot path.
    """

    def __init__(
        self,
        input_data: torch.Tensor,
        *,
        patch_shape: tuple[int, ...],
        patch_overlap: tuple[int, ...],
        noise_map=None,
        mask: torch.Tensor | NDArray | None = None,
        mask_threshold=50,
    ):
        device = input_data.device
        data_shape = input_data.shape
        if mask is None:
            mask = torch.ones(data_shape[:-1], dtype=torch.float32, device=device)
        else:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(dtype=torch.float32, device=device)
        if mask.shape == data_shape[:-1]:  # only spatial mask provided
            mask = mask[..., None].expand(data_shape).contiguous()

        self._starts = patch_grid(data_shape, patch_shape, patch_overlap, device)
        self.grid_shape = tuple(len(st) for st in self._starts)
        self.patch_locs = select_patches_to_process(
            mask, patch_shape, self._starts, mask_threshold
        )
        self.mask = mask
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        self.input_data = input_data
        self._strides = torch.tensor(
            input_data.stride(), dtype=torch.int64, device=device
        )

        # Per-selected-patch noise variance
        self.var_apriori_by_patch = None
        if noise_map is not None:
            if isinstance(noise_map, (float, np.floating)):
                noise_map = torch.full(
                    data_shape, float(noise_map), dtype=torch.float32, device=device
                )
            else:
                if isinstance(noise_map, np.ndarray):
                    noise_map = torch.from_numpy(noise_map)
                noise_map = noise_map.to(dtype=torch.float32, device=device)
                if noise_map.shape == data_shape[:-1]:
                    noise_map = noise_map[..., None].expand(data_shape).contiguous()
            var_sums = sliding_sum_nd(noise_map**2, patch_shape, self._starts)
            self.var_apriori_by_patch = var_sums.ravel()[self.patch_locs] / np.prod(
                patch_shape
            )

    def __len__(self):
        """Get number of patches to process."""
        return len(self.patch_locs)

    def _top_left(self, start: int, stop: int) -> torch.Tensor:
        """Top-left corner indices of patches ``[start, stop)``, ``(B, ndim)``."""
        grid_idx = torch.unravel_index(self.patch_locs[start:stop], self.grid_shape)
        return torch.stack([st[g] for st, g in zip(self._starts, grid_idx)], dim=-1)

    def get_batch(self, start: int, stop: int):
        """Gather patches ``[start, stop)`` and their top-left corner indices.

        A single vectorized advanced-index gather over the (already
        GPU-resident) input data -- no per-patch Python loop, no host-device
        copy.
        """
        top_left_idx = self._top_left(start, stop)
        ndim = len(self.patch_shape)
        index = tuple(
            (
                top_left_idx[:, d, None] + torch.arange(ps, device=top_left_idx.device)
            ).view(-1, *(ps if j == d else 1 for j in range(ndim)))
            for d, ps in enumerate(self.patch_shape)
        )
        return self.input_data[index], top_left_idx

    def get_offsets(self, start: int, stop: int):
        """Element offsets and top-left corner indices of patches ``[start, stop)``.

        The offsets locate each patch's first voxel in the input data, for
        kernels that read patches in place instead of gathering them.
        """
        top_left_idx = self._top_left(start, stop)
        return (top_left_idx * self._strides).sum(-1), top_left_idx
