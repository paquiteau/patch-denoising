"""Common utilities for bindings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from patch_denoise.denoise import (
    adaptive_thresholding,
    hybrid_pca,
    mp_pca,
    nordic,
    optimal_thresholding,
    raw_svt,
)

DENOISER_MAP = {
    None: None,
    "mp-pca": mp_pca,
    "hybrid-pca": hybrid_pca,
    "raw": raw_svt,
    "optimal-fro": lambda *args, **kwargs: optimal_thresholding(
        *args, loss="fro", **kwargs
    ),
    "optimal-fro-noise": lambda *args, **kwargs: optimal_thresholding(
        *args, loss="fro", **kwargs
    ),
    "optimal-nuc": lambda *args, **kwargs: optimal_thresholding(
        *args, loss="nuc", **kwargs
    ),
    "optimal-ope": lambda *args, **kwargs: optimal_thresholding(
        *args, loss="ope", **kwargs
    ),
    "nordic": nordic,
    "adaptive-qut": lambda *args, **kwargs: adaptive_thresholding(
        *args, method="qut", **kwargs
    ),
}

_RECOMBINATION = {"w": "weighted", "c": "center", "a": "average"}


@dataclass
class DenoiseParameters:
    """Denoise Parameters data structure."""

    method: str = None
    patch_shape: int | tuple[int, ...] = 11
    patch_overlap: int | tuple[int, ...] = 0
    recombination: str = "weighted"  # "center" is also available
    mask_threshold: int = 10

    @property
    def pretty_name(self):
        """Return a pretty name for the representation of parameters."""
        if self.method:
            name = self.method
            for attr in [
                "patch_shape",
                "patch_overlap",
                "recombination",
                "mask_threshold",
            ]:
                if getattr(self, attr):
                    name += f"_{getattr(self, attr)}"
        else:
            name = "noisy"
        return name

    @property
    def pretty_par(self):
        """Get pretty representation of parameters."""
        name = f"{self.patch_shape}_{self.patch_overlap}{self.recombination[0]}"
        return name

    @classmethod
    def get_str(cls, **kwargs):
        """Get full string representation from set of kwargs."""
        return cls(**kwargs).pretty_name

    @classmethod
    def from_str(self, config_str):
        """Create a DenoiseParameters from a string."""
        if "noisy" in config_str:
            return DenoiseParameters(
                method=None,
                patch_shape=None,
                patch_overlap=None,
                recombination=None,
                mask_threshold=None,
            )
        else:
            conf = config_str.split("_")
            d = DenoiseParameters()
            if conf:
                d.method = conf.pop(0)
            if conf:
                d.patch_shape = int(conf.pop(0))
            if conf:
                d.patch_overlap = int(conf.pop(0))
            if conf:
                c = conf.pop(0)
                d.recombination = c
            if conf:
                d.mask_threshold = int(conf.pop(0))
            return d

    def __str__(self):
        """Get string representation."""
        return self.pretty_name


def load_as_array(input):
    """Load a file as a numpy array, and return affine matrix if available."""
    import nibabel as nib

    if input is None:
        return None, None
    if input.suffix == ".npy":
        return np.load(input), None
    elif ".nii" in input.suffixes:
        nii = nib.load(input)
        return nii.get_fdata(dtype=np.float32), nii.affine
    else:
        raise ValueError("Unsupported file format. use numpy or nifti formats.")


def save_array(data, affine, filename):
    """Save array to file, with affine matrix if required."""
    import nibabel as nib

    if filename is None:
        return None

    if ".nii" in filename.suffixes:
        if affine is None:
            affine = np.eye(len(data.shape))
        nii_img = nib.Nifti1Image(data, affine)
        nii_img.to_filename(filename)
    elif filename.endswith(".npy"):
        np.save(filename, data)

    return filename


def load_complex_nifti(mag_file, phase_file, filename=None):  # pragma: no cover
    """Load two nifti image (magnitude and phase) to create a complex valued array.

    Optionally, the result can be save as a .npy file

    Parameters
    ----------
    mag_file: str
        The source magnitude file
    phase_file: str
        The source phase file
    filename: str, default None
        The output filename
    """
    mag, mag_affine = load_as_array(mag_file)
    phase, phase_affine = load_as_array(phase_file)

    if not np.allclose(mag_affine, phase_affine):
        logging.warning("Affine matrices for magnitude and phase are not the same")

    logging.info("Phase data range is [%.2f %.2f]", np.min(phase), np.max(phase))
    logging.info("Mag data range is [%.2f %.2f]", np.min(mag), np.max(mag))
    img = mag * np.exp(1j * phase)

    if filename is not None:
        np.save(filename, img)
    return img, mag_affine


def compute_mask(array, convex=False):
    """Compute mask for array using the Otzu's method.

    The time axis is assumed to be the last one.

    The mask is computed slice-wise on the time average of the array.

    Parameters
    ----------
    array : numpy.ndarray
        Array to compute mask for.
    convex : bool, default False
        If True, the mask is convex for each slice.

    Returns
    -------
    numpy.ndarray
        Mask for array.
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import convex_hull_image

    mean = array.mean(axis=-1)
    mask = np.zeros(mean.shape, dtype=bool)
    for i in range(mean.shape[-1]):
        mask[..., i] = mean[..., i] > threshold_otsu(mean[..., i])
    if convex:
        for i in range(mean.shape[-1]):
            mask[..., i] = convex_hull_image(mask[..., i])
    return mask
