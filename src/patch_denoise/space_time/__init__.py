"""Denoising Operator working on space-time dimension."""

__all__ = [
    "BaseSpaceTimeDenoiser",
    "MPPCADenoiser",
    "HybridPCADenoiser",
    "NordicDenoiser",
    "OptimalSVDDenoiser",
    "RawSVDDenoiser",
]

_LOWRANK = {
    "MPPCADenoiser",
    "HybridPCADenoiser",
    "NordicDenoiser",
    "OptimalSVDDenoiser",
    "RawSVDDenoiser",
}


def __getattr__(name: str):
    """Lazily import denoisers, so ``import patch_denoise.space_time`` stays light."""
    if name == "BaseSpaceTimeDenoiser":
        from .base import BaseSpaceTimeDenoiser

        return BaseSpaceTimeDenoiser
    if name in _LOWRANK:
        from . import lowrank

        return getattr(lowrank, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
