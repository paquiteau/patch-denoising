"""
Experimental Data denoising
===========================

This is a example script to test various denoising methods on real-word fMRI data.

Source data should a sequence of 2D or 3D data, the temporal dimension being the last one.

The available denoising methods are "nordic", "mp-pca", "hybrid-pca", "opt-fro", "opt-nuc" and "opt-op".
"""

import nibabel as nib
from patch_denoise.space_time.lowrank import OptimalSVDDenoiser
import timeit

# %%
# Setup the parameters for the simulation and noise

SHAPE = (64, 64, 64)
N_FRAMES = 200

NOISE_LEVEL = 2

input_path = "/data/parietal/store2/data/ibc/3mm/sub-01/ses-00/func/wrdcsub-01_ses-00_task-ArchiSocial_dir-ap_bold.nii.gz"
output_path = "/scratch/ymzayek/retreat_data/output.nii"

img = nib.load(input_path)

# data shape is (53, 63, 52, 262) with 3mm resolution
patch_shape = (11, 11, 11)
patch_overlap = (5)

# initialize denoiser
optimal_llr = OptimalSVDDenoiser(patch_shape, patch_overlap)

# denoise image
time_start = timeit.default_timer()
denoised = optimal_llr.denoise(img.get_fdata())
print(timeit.default_timer() - time_start)