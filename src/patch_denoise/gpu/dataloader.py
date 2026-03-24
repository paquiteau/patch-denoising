"""Torch dataloader for the noisy data"""

import torch
import numpy as np
from patch_denoise.space_time.base import PatchedArray


class PatchDataset(torch.utils.data.Dataset):
    """Dataset for extracting patches from the input data."""

    def __init__(
        self,
        input_data,
        *,
        patch_shape,
        patch_overlap,
        noise_map=None,
        mask=None,
        mask_threshold=0.5,
    ):
        # Leverage the patchedArray interface from CPU version
        # TODO: this can be a bit memory demanding on the cpu side - consider implementing a more memory efficient version (e.g. memmap backed patches)

        data_shape = input_data.shape
        self.input_data = PatchedArray(
            input_data, patch_shape=patch_shape, patch_overlap=patch_overlap
        )

        if mask is None:
            process_mask = np.full(data_shape, dtype=torch.bool)
        elif mask.shape == data_shape[:-1]:  # only spatial mask provided
            process_mask = np.broadcast_to(mask[..., None], data_shape).astype(bool)

        process_mask = PatchedArray(
            process_mask,
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
            paddin_mode="constant",
            constant_value=0,
        )

        self.select_patch = [
            i
            for i in range(self.input_data.n_patches)
            if 100 * process_mask.get_patch(i).mean() < mask_threshold
        ]

    def __len__(self):
        """Get number of patches to process."""
        return len(self.select_patch)

    def __getitem__(self, idx):
        """Get the patch and its corresponding indices."""
        patch_idx = self.select_patch[idx]
        patch_data = self.input_data.get_patch(patch_idx)

        return patch_data, patch_idx
