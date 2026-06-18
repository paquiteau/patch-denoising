"""Collection of patch-based denoising methods."""

from importlib.metadata import PackageNotFoundError, version

from patch_denoise.denoise import (
    adaptive_thresholding,
    hybrid_pca,
    mp_pca,
    nordic,
    optimal_thresholding,
    raw_svt,
)
from patch_denoise.space_time.lowrank import (
    AdaptiveDenoiser,
    HybridPCADenoiser,
    MPPCADenoiser,
    NordicDenoiser,
    OptimalSVDDenoiser,
    RawSVDDenoiser,
)

__all__ = [
    "AdaptiveDenoiser",
    "HybridPCADenoiser",
    "MPPCADenoiser",
    "NordicDenoiser",
    "OptimalSVDDenoiser",
    "RawSVDDenoiser",
    "mp_pca",
    "hybrid_pca",
    "optimal_thresholding",
    "adaptive_thresholding",
    "raw_svt",
    "nordic",
]


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
