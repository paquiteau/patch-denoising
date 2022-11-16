"""Common utilities for bindings."""

from denoiser.denoise import (
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
