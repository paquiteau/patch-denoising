"""
Visualisation of spatio-temporal data
=====================================

This is a example script to test various denoising methods on real-word fMRI data.

Source data should a sequence of 2D or 3D data, the temporal dimension being the last one.

"""
import matplotlib.pyplot as plt

from patch_denoise.simulation.phantom import mr_shepp_logan_t2_star, g_factor_map
from patch_denoise.simulation.activations import add_frames
from patch_denoise.simulation.noise import add_temporal_gaussian_noise

# %%
# Setup the parameters for the simulation and noise

SHAPE = (64, 64, 64)
N_FRAMES = 400

NOISE_LEVEL = 2

# %%
# Simulate data for the reconstruction. We use a classical Shepp-Logan phantom, on
# which we add gaussian noise.
# Note that the simulation data is real-valued, for a more pratical case check the
# ``experimental_data.py`` file.

# Create a 2D phantom.
phantom = mr_shepp_logan_t2_star(SHAPE)[32]
ground_truth = add_frames(phantom, N_FRAMES)
g_map = g_factor_map(SHAPE)
print(g_map.shape)

noisy_image = add_temporal_gaussian_noise(ground_truth, sigma=NOISE_LEVEL)

# %%
# Unrolling using carpetplot
# --------------------------
# Any spatio temporal data can be unrolled like a carpet to show the temporal evolution
# of every voxel.

from patch_denoise.viz.plots import carpet_plot


carpet_plot(noisy_image[32:48, 32:48], unfold="classic", transpose=True)

# %%
# In case of 2D data the voxels can be unfolded using a zig-zag pattern.
carpet_plot(noisy_image[32:48, 32:48], unfold="zigzag")
