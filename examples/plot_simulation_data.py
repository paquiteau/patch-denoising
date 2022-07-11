# -*- coding: utf-8 -*-
"""
Experimental Data denoising
===========================

This is a example script to test various denoising methods on real-word fMRI data.

Source data should a sequence of 2D or 3D data, the temporal dimension being the last one.

The available denoising methods are "nordic", "mp-pca", "hybrid-pca", "opt-fro", "opt-nuc" and "opt-op".

"""
import numpy as np
import matplotlib.pyplot as plt


from denoiser.data.phantom import mr_shepp_logan_t2_star, g_factor_map
from denoiser.data.activations import add_activations
from denoiser.data.noise import add_temporal_gaussian_noise

#%%
# Setup the parameters for the simulation and noise

DIM = 3
SHAPE = (64, 64, 64)
N_FRAMES = 40

NOISE_LEVEL = 2


#%%
# Simulate data for the reconstruction. We use a classical Shepp-Logan phantom, on
# which we add gaussian noise.
# Note that the simulation data is real-valued, for a more pratical case check the
# ``experimental_data.py`` file.


phantom = mr_shepp_logan_t2_star(SHAPE)
ground_truth = add_activations(phantom, N_FRAMES)
g_map = g_factor_map(SHAPE)
print(g_map.shape)
plt.figure()
plt.subplot(241)
plt.imshow(ground_truth[SHAPE[0] // 2, :, :, 0])
plt.subplot(242)
plt.imshow(ground_truth[:, SHAPE[1] // 2, :, 0])
plt.subplot(243)
plt.imshow(ground_truth[:, :, SHAPE[2] // 2, 0])
plt.subplot(244)
plt.imshow(g_map[:, :, SHAPE[2] // 2])

noisy_image = add_temporal_gaussian_noise(ground_truth, sigma=NOISE_LEVEL)

plt.subplot(245)
plt.imshow(noisy_image[SHAPE[0] // 2, :, :, 0])
plt.subplot(246)
plt.imshow(noisy_image[:, SHAPE[1] // 2, :, 0])
plt.subplot(247)
plt.imshow(noisy_image[:, :, SHAPE[2] // 2, 0])
plt.tight_layout()
plt.show()

#%%
# Perform denoising
#
