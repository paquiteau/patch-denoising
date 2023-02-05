"""Common utilities for bindings."""
from dataclasses import dataclass

from patch_denoise.denoise import (
    hybrid_pca,
    mp_pca,
    nordic,
    optimal_thresholding,
    raw_svt,
    adaptive_thresholding,
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
                d.recombination = "weighted" if c == "w" else "center"
            if conf:
                d.mask_threshold = conf.pop(0)
            return d
