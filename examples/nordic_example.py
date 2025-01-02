"""
===========================================================
Apply NORDIC to complex-valued fMRI data with noise volumes
===========================================================

This example shows how to apply NORDIC to complex-valued fMRI data with noise
volumes.
"""

###############################################################################
# Load Data
# -----------------------------------------------------------------------------

import nibabel as nb
import numpy as np

magnitude_file = 'data/sub-01_task-rest_part-mag_bold.nii.gz'
phase_file = 'data/sub-01_task-rest_part-phase_bold.nii.gz'
magnitude_noise_file = 'data/sub-01_task-rest_part-mag_noRF.nii.gz'
phase_noise_file = 'data/sub-01_task-rest_part-phase_noRF.nii.gz'

magnitude_img = nb.load(magnitude_file)
phase_img = nb.load(phase_file)
magnitude_noise_img = nb.load(magnitude_noise_file)
phase_noise_img = nb.load(phase_noise_file)

magnitude_data = magnitude_img.get_fdata()
phase_data = phase_img.get_fdata()
magnitude_noise_data = magnitude_noise_img.get_fdata()
phase_noise_data = phase_noise_img.get_fdata()

n_noise_volumes = magnitude_noise_data.shape[3]

magnitude_data = np.concatenate((magnitude_data, magnitude_noise_data), axis=3)
phase_data = np.concatenate((phase_data, phase_noise_data), axis=3)

# Rescale phase data to [-pi, pi]
phase_range = np.max(phase_data)
phase_range_min = np.min(phase_data)
range_norm = phase_range - phase_range_min
range_center = (phase_range + phase_range_min) / range_norm * 1 / 2
phase_data = (phase_data / range_norm - range_center) * 2 * np.pi

# Combine magnitude and phase into complex-valued data
complex_data = magnitude_data * np.exp(1j * phase_data)

# TODO: Filter complex data and calculate g-factor map

###############################################################################
# Calculate Noise Level
# -----------------------------------------------------------------------------

# Split out the noise volumes
noise_data = complex_data[:, :, :, -n_noise_volumes:]
noise_data[np.isnan(noise_data)] = 0
noise_data[np.isinf(noise_data)] = 0
noise_level = np.std(noise_data[noise_data != 0])

# Scale by sqrt(2) because the data are complex-valued
noise_level = noise_level / np.sqrt(2)

###############################################################################
# Run NORDIC
# -----------------------------------------------------------------------------
from patch_denoise.denoise import nordic

n_vols = complex_data.shape[3]
patch_shape = np.ones(3, dtype=int) * int(np.round(np.cbrt(n_vols * 11)))
denoised_data, patch_weights, noise, dofs = nordic(
    input_data=complex_data,
    patch_shape=patch_shape,
    patch_overlap=2,
    noise_std=noise_level,
    recombination='average',
    n_iter_threshold=10,
)
denoised_data = denoised_data[:, :, :, :-n_noise_volumes]
denoised_magnitude_data = np.abs(denoised_data)
denoised_phase_data = np.angle(denoised_data)
denoised_magnitude_img = nb.Nifti1Image(
    denoised_magnitude_data,
    magnitude_img.affine,
    magnitude_img.header,
)
denoised_phase_img = nb.Nifti1Image(
    denoised_phase_data,
    phase_img.affine,
    phase_img.header,
)

dofs_img = nb.Nifti1Image(
    dofs,
    magnitude_img.affine,
    magnitude_img.header,
)
