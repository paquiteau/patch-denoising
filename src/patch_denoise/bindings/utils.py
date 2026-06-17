"""Common utilities for bindings."""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from patch_denoise.denoise import (
    adaptive_thresholding,
    hybrid_pca,
    mp_pca,
    nordic,
    optimal_thresholding,
    raw_svt,
)

DENOISER_MAP = {
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


def load_as_array(input: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a file as a numpy array, and return affine matrix if available."""
    import nibabel as nib

    if input is None:
        return None, None
    if input.suffix == ".npy":
        return np.load(input), np.eye(4)
    elif ".nii" in input.suffixes:
        nii = nib.Nifti1Image.from_filename(input)
        return nii.get_fdata(dtype=np.float32), nii.affine
    else:
        raise ValueError("Unsupported file format. use numpy or nifti formats.")


def save_array(data: NDArray, affine: NDArray, filename: Path) -> Path:
    """Save array to file, with affine matrix if required."""
    if ".nii" in filename.suffixes:
        if affine is None:
            affine = np.eye(len(data.shape))
        nii_img = nib.Nifti1Image(data, affine)
        nii_img.to_filename(filename)
    elif filename.suffix == ".npy":
        np.save(filename, data)

    return filename


def load_complex_nifti(
    mag_file: Path, phase_file: Path
) -> tuple[NDArray, NDArray]:  # pragma: no cover
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

    return img, mag_affine


def fast_cuda_check() -> bool:
    """
    Instantly checks if PyTorch is installed and CUDA is supported by the system.

    Without paying the heavy time/memory cost of 'import torch'.
    """
    # 1. Instant check: Is PyTorch even installed in this environment? (~1ms)
    if importlib.util.find_spec("torch") is None:
        return False

    # 2. Fast check: Does an NVIDIA GPU driver exist on the system? (~0ms)
    if shutil.which("nvidia-smi") is None:
        return False

    # 3. Quick check: Is the GPU actually responsive/functional? (~10-20ms)
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
