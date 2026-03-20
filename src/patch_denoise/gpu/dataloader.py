"""Torch dataloader for the noisy data"""

import torch
from patch_denoise.space_time.base import PatchedArray

class PatchDataset(torch.utils.data.Dataset):
    """Dataset for extracting patches from the input data."""

    def __init__(
            self, input_data,*, patch_shape, patch_overlap, noise_map=None, mask=None, mask_threshold=0.5
    ):
        # Leverage the patchedArray interface from CPU version
        # TODO: this can be a bit memory demanding on the cpu side - consider implementing a more memory efficient version (e.g. memmap backed patches)
        
        
    def __getitem__(self, idx):
        # return patch and its corresponding indices
        
