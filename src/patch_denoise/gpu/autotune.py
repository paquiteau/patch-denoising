"""Auto-tune the GPU batch size for a given denoising configuration.

The best batch size depends on the GPU, the denoising method and the patch
shape in ways that aren't predictable from hardware specs alone (see the
benchmarks that motivated this: the sweet spot moved from ~96-192 for
(11,11,11) patches to a flat plateau across 64-1024 for (5,5,5) patches on
the same GPU). Instead of guessing, this times a short forward pass at a
handful of candidate sizes on the real GPU/method/patch shape and picks the
fastest, then caches the result to disk so repeat runs skip calibration.
"""

import json
import logging
import time
from pathlib import Path

import torch

log = logging.getLogger(__name__)

CACHE_PATH = Path.home() / ".cache" / "patch_denoise" / "gpu_batch_size.json"
CANDIDATE_BATCH_SIZES = (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
N_WARMUP = 2
N_TRIALS = 5


def _cache_key(method: str, patch_shape: tuple[int, ...], recombination: str) -> str:
    gpu_name = torch.cuda.get_device_name(0)
    return f"{gpu_name}|{method}|{tuple(patch_shape)}|{recombination}"


def _load_cache() -> dict[str, int]:
    try:
        return json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, int]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def autotune_batch_size(
    method: str,
    patch_shape: tuple[int, int, int, int],
    recombination: str,
    candidates: tuple[int, ...] = CANDIDATE_BATCH_SIZES,
    n_trials: int = N_TRIALS,
    **denoiser_kwargs,
) -> int:
    """Return the fastest batch size for this (GPU, method, patch_shape).

    Cached to disk keyed by GPU model + method + patch shape + recombination;
    a repeat run with the same configuration reuses the cached value instead
    of re-measuring.
    """
    from .main import make_denoiser

    key = _cache_key(method, patch_shape, recombination)
    cache = _load_cache()
    if key in cache:
        log.info(f"Using cached GPU batch size {cache[key]} for {key!r}.")
        return cache[key]

    log.info(
        "Auto-tuning GPU batch size for this configuration (one-time, cached "
        f"to {CACHE_PATH})..."
    )
    best_bs, best_us_per_patch = candidates[0], float("inf")
    for bs in candidates:
        try:
            denoiser = make_denoiser(
                method,
                patch_shape=patch_shape,
                recombination=recombination,
                batch_size=bs,
                **denoiser_kwargs,
            )
            dummy = torch.randn(bs, *patch_shape, device="cuda", dtype=torch.float32)
            with torch.inference_mode():
                for _ in range(N_WARMUP):
                    denoiser(dummy)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_trials):
                    denoiser(dummy)
                torch.cuda.synchronize()
                us_per_patch = (time.perf_counter() - t0) / (n_trials * bs) * 1e6
            log.debug(f"  batch_size={bs}: {us_per_patch:.1f} us/patch")
            if us_per_patch < best_us_per_patch:
                best_us_per_patch = us_per_patch
                best_bs = bs
        except torch.OutOfMemoryError:
            log.debug(f"  batch_size={bs}: out of memory, stopping search here.")
            torch.cuda.empty_cache()
            break
        finally:
            torch.cuda.empty_cache()

    log.info(
        f"Auto-tuned GPU batch size: {best_bs} ({best_us_per_patch:.1f} us/patch)."
    )
    cache[key] = best_bs
    _save_cache(cache)
    return best_bs
