"""Common utilities for bindings."""
from dataclasses import dataclass

import numpy as np

from patch_denoise.denoise import (
    hybrid_pca,
    mp_pca,
    nordic,
    optimal_thresholding,
    raw_svt,
    adaptive_thresholding,
)

NIBABEL_AVAILABLE = True
try:
    import nibabel as nib
except ImportError:
    NIBABEL_AVAILABLE = False

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
    patch_shape: int = 11
    patch_overlap: int = 0
    recombination: str = "weighted"  # "center" is also available
    mask_threshold: int = 10

    @classmethod
    def from_str(self, config_str):
        """Parse config string to create data structure.

        The full format is ::

            <method>_<patch_shape>_<patch_overlap>_<recombination>_<mask_threshold>
        """
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
                try:
                    d.recombination = _RECOMBINATION[c]
                except KeyError as exc:
                    raise ValueError(f"unsupported  recombination key: {c}") from exc
            if conf:
                d.mask_threshold = int(conf.pop(0))
            return d

    def __str__(self):
        """Return string reprensation."""
        ret_str = f"{self.method}_{self.patch_shape}_{self.patch_overlap}_"
        ret_str += self.recombination[0]
        if self.mask_threshold:
            ret_str += f"_{self.mask_threshold}"
        return ret_str


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
    if not NIBABEL_AVAILABLE:
        raise RuntimeError(
            "nibabel is not available, please install it to load experimental data"
        )

    mag = nib.load(mag_file).get_fdata()
    phase = nib.load(phase_file).get_fdata()
    print(np.min(phase), np.max(phase))
    print(np.min(mag), np.max(mag))
    img = mag * np.exp(1j * phase)

    if filename is not None:
        np.save(filename, img)
    return img
