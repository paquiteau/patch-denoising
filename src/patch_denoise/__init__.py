"""Collection of patch-based denoising methods."""

from importlib.metadata import PackageNotFoundError, version

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

_FUNCTIONAL = {
    "mp_pca",
    "hybrid_pca",
    "optimal_thresholding",
    "adaptive_thresholding",
    "raw_svt",
    "nordic",
}
_DENOISERS = {
    "AdaptiveDenoiser",
    "HybridPCADenoiser",
    "MPPCADenoiser",
    "NordicDenoiser",
    "OptimalSVDDenoiser",
    "RawSVDDenoiser",
}


def __getattr__(name: str):
    """Lazily import denoisers, so ``import patch_denoise`` stays lightweight."""
    if name in _FUNCTIONAL:
        from patch_denoise import denoise

        return getattr(denoise, name)
    if name in _DENOISERS:
        from patch_denoise.space_time import lowrank

        return getattr(lowrank, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
