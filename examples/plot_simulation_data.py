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


from denoiser.simulation.phantom import mr_shepp_logan_t2_star, g_factor_map
from denoiser.simulation.activations import add_activations
from denoiser.simulation.noise import add_temporal_gaussian_noise

#%%
# Setup the parameters for the simulation and noise

SHAPE = (64, 64, 64)
N_FRAMES = 400

NOISE_LEVEL = 2


#%%
# Simulate data for the reconstruction. We use a classical Shepp-Logan phantom, on
# which we add gaussian noise.
# Note that the simulation data is real-valued, for a more pratical case check the
# ``experimental_data.py`` file.


phantom = mr_shepp_logan_t2_star(SHAPE)[32]
ground_truth = add_activations(phantom, N_FRAMES)
g_map = g_factor_map(SHAPE)
print(g_map.shape)

noisy_image = add_temporal_gaussian_noise(ground_truth, sigma=NOISE_LEVEL)

#%%
# An Example of visualizatoin using carpet plot.

from denoiser.viz.plots import carpet_plot


carpet_plot(noisy_image, unfold="classic")
#%%

carpet_plot(noisy_image, unfold="zigzag")
