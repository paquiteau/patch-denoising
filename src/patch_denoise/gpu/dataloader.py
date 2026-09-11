"""Torch dataloader for the noisy data."""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


def patchify_tensor(data, patch_shape, patch_overlap):
    """
    Transform a tensor into a collection of patches with specified shape and overlap.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor.
    patch_shape : tuple of int
        The shape of each patch.
    patch_overlap : tuple of int
        The number of overlapping elements between adjacent patches along each dimension

    Returns
    -------
    torch.Tensor
        A view of the original tensor containing the extracted patches, with
        shape (grid_patches, *patch_shape).

    """
    _ps = np.array(patch_shape)
    _po = np.array(patch_overlap)
    dimensions = data.ndim

    step = _ps - _po
    if np.any(step < 0):
        raise ValueError("overlap should be smaller than patch on every dimension.")

    if _ps.size != dimensions or step.size != dimensions:
        raise ValueError(
            "_ps and step must have the same number of dimensions as the input _array."
        )

    # Ensure patch size is not larger than _array size along each axis
    _ps = np.minimum(_ps, data.shape)

    # Calculate the shape and strides of the sliding view
    grid_shape = tuple(
        (((data.shape[i] - _ps[i]) // step[i] + 1) if _ps[i] < data.shape[i] else 1)
        for i in range(dimensions)
    )
    shape = grid_shape + tuple(_ps)
    strides = (
        tuple(
            (data.stride()[i] * step[i] if _ps[i] < data.shape[i] else 0)
            for i in range(dimensions)
        )
        + data.stride()
    )
    return torch.as_strided(data, shape, strides)


def sliding_sum_nd(data, patch_shape, patch_overlap):
    """
    Compute a sliding sum across all dimensions using sequential 1D convolutions.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor.
    patch_shape : tuple of int
        The shape of each patch.
    patch_overlap : tuple of int
        The number of overlapping elements between adjacent patches along each dimension

    Returns
    -------
    torch.Tensor
        results of the sliding sum, with the same grid shape as the output of
        patchify_tensor, but with the values being the sum of the corresponding
        patches in the input data.

    Notes
    -----
    Here a convolution-based approach is used to efficiently compute the sliding
    sum over each dimension (problem is separable). This is (significantly) more
    memory efficient than working on the patchified tensor.
    """
    res = data
    patch_step = tuple(s - o for s, o in zip(patch_shape, patch_overlap))
    for dim in range(res.ndim):
        k = patch_shape[dim]
        s = patch_step[dim]

        if k <= 1:
            continue  # A patch of size 1 sums to the data itself, nothing to do

        # Move the target dimension to the last position, other dims are seen as batch
        curr_shape = res.shape
        res = res.movedim(dim, -1)
        res_shape_moved = res.shape
        res = res.reshape(-1, 1, curr_shape[dim])

        weight = torch.ones((1, 1, k), dtype=res.dtype, device=res.device)
        res = F.conv1d(res, weight, stride=s if s > 0 else k)

        new_dim_len = res.shape[-1]
        new_shape = list(res_shape_moved)
        new_shape[-1] = new_dim_len

        res = res.view(new_shape).movedim(-1, dim)
    return res


def select_patches_to_process(mask, patch_shape, patch_overlap, mask_threshold=50):
    """Select patches to process based on the mask and threshold.

    Parameters
    ----------
    mask : numpy.ndarray
        The input mask, with the same spatial dimensions as the input data.
    patch_shape : tuple of int
        The shape of each patch.
    patch_overlap : tuple of int
        The number of overlapping elements between adjacent patches along each dimension
    mask_threshold : float, default 50
        The percentage threshold for selecting patches. Patches with a
        percentage of masked pixels above this threshold will be selected for
        processing.

    Returns
    -------
    list of int
        A list of indices corresponding to the selected patches to process.
    """
    # move to cuda to be super fast
    with torch.inference_mode():
        mask_g = mask.to(dtype=torch.float32, device="cuda")
        patch_score_g = sliding_sum_nd(mask_g, patch_shape, patch_overlap)
        patch_score_g /= np.prod(patch_shape)

        patch_score = patch_score_g.cpu().ravel()
        patch_idxs = torch.where(patch_score > mask_threshold / 100)[0]

    del mask_g, patch_score_g
    return patch_idxs


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

        self.patch_locs = select_patches_to_process(
            mask, patch_shape, patch_overlap, mask_threshold
        ).to(device)
        self.mask = mask
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        self._step = torch.tensor(
            [ps - po for ps, po in zip(patch_shape, patch_overlap)],
            dtype=torch.int64,
            device=device,
        )

        self.input_data = patchify_tensor(input_data, patch_shape, patch_overlap)
        self.grid_shape = self.input_data.shape[: len(data_shape)]

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
            var_patches = patchify_tensor(noise_map**2, patch_shape, patch_overlap)
            grid_idx = torch.unravel_index(self.patch_locs, self.grid_shape)
            selected_var_patches = var_patches[grid_idx]
            self.var_apriori_by_patch = selected_var_patches.mean(
                dim=tuple(range(1, selected_var_patches.ndim))
            )

    def __len__(self):
        """Get number of patches to process."""
        return len(self.patch_locs)

    def get_batch(self, start: int, stop: int):
        """Gather patches ``[start, stop)`` and their top-left corner indices.

        A single vectorized advanced-index gather over the (already
        GPU-resident) strided patch view -- no per-patch Python loop, no
        host-device copy.
        """
        grid_idx = torch.unravel_index(self.patch_locs[start:stop], self.grid_shape)
        patch_data = self.input_data[grid_idx]
        top_left_idx = torch.stack(grid_idx, dim=-1) * self._step
        return patch_data, top_left_idx
