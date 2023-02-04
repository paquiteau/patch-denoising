"""
Experimental Data denoising
===========================

This is a example script to test various denoising methods on real-word fMRI data.

Source data should a sequence of 2D or 3D data, the temporal dimension being the last one.

The available denoising methods are "nordic", "mp-pca", "hybrid-pca", "opt-fro", "opt-nuc" and "opt-op".
"""

from denoiser.simulation.phantom import mr_shepp_logan_t2_star, g_factor_map
from denoiser.simulation.activations import add_frames
from denoiser.simulation.noise import add_temporal_gaussian_noise

# %%
# Setup the parameters for the simulation and noise

SHAPE = (64, 64, 64)
N_FRAMES = 200

NOISE_LEVEL = 2
