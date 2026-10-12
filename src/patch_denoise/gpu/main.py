"""Main loop for the gpu version of patch-denoise."""

import logging
from functools import partial

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.rich import tqdm

from .._docs import fill_doc
from ..space_time.base import (
    DenoiserName,
    ExtraOutput,
    Recombination,
    _patch_param,
    center_padding,
    check_center_recombination_overlap,
)
from ._accumulators import launch_accumulate_recon, launch_broadcast_add
from ._svd import fused_supported
from .autotune import autotune_batch_size
from .dataloader import PatchDataset
from .denoiser import MPPCADenoiser, OptimalSVDDenoiser

log = logging.getLogger(__name__)

_NEEDS_COUNT = ExtraOutput.COUNT | ExtraOutput.NOISE_STD | ExtraOutput.RANK
_NO_EXTRA_OUTPUT = ExtraOutput(0)

# Single source of truth for GPU-supported methods, mirroring make_denoiser's
# own dispatch below -- callers (e.g. the CLI) should check membership here
# instead of hardcoding their own method list that can drift out of sync.
GPU_SUPPORTED_METHODS = frozenset(
    m for m in DenoiserName if "optimal" in m or m == DenoiserName.MP_PCA
)


def _uses_fused(dtype, patch_shape) -> bool:
    """Whether main_gpu reads patches in place, see the main loop."""
    return fused_supported(dtype, patch_shape[-1])


def _dummy_run(denoiser, bs, patch_shape, dtype, fused):
    """Return a callable denoising ``bs`` random patches the way main_gpu would."""
    if not fused:
        dummy = torch.randn(bs, *patch_shape, device="cuda", dtype=dtype)
        return partial(denoiser, dummy)
    # Disjoint patches stacked along the first axis of a volume.
    px, *rest = patch_shape
    vol = torch.randn(bs * px, *rest, device="cuda", dtype=dtype)
    bases = torch.arange(bs, device="cuda") * (px * vol.stride(0))
    out, wsum = torch.zeros_like(vol), torch.zeros_like(vol)
    if denoiser.recombination == Recombination.CENTER:
        return partial(denoiser.forward_center, vol, bases)
    return partial(denoiser.forward_patches, vol, bases, out, wsum)


def make_denoiser(
    method,
    patch_shape,
    recombination,
    batch_size,
    full_time=False,
    dtype=torch.float32,
    **kwargs,
) -> OptimalSVDDenoiser | MPPCADenoiser:
    """Create a denoiser model on GPU."""
    if "optimal" in method:
        # method is "optimal-{loss}" or "optimal-{loss}-noise" ("-noise" only
        # signals that a noise map is supplied at forward time, see
        # main_gpu's var_apriori handling; it isn't part of the loss name).
        loss = method.removeprefix("optimal-").removesuffix("-noise")
        denoiser = OptimalSVDDenoiser(
            patch_shape=patch_shape,
            recombination=recombination,
            loss=loss,
            full_time=full_time,
            **kwargs,
        )
    elif method == "mp-pca":
        denoiser = MPPCADenoiser(
            patch_shape=patch_shape,
            recombination=recombination,
            full_time=full_time,
            **kwargs,
        )
    else:
        raise ValueError(f"method {method} is not supported on GPU. ")

    denoiser = denoiser.cuda()  # Move model to GPU

    # Warm up: builds FastPatchSVD's cuSOLVER workspace and JIT-compiles its
    # Triton kernels for this batch size before the tracked loop starts, so
    # construction errors fail fast instead of surfacing on the first batch.
    with torch.inference_mode():
        fused = _uses_fused(dtype, patch_shape)
        _dummy_run(denoiser, batch_size, patch_shape, dtype, fused)()
    torch.cuda.empty_cache()

    return denoiser


@torch.inference_mode()
@fill_doc
def main_gpu(
    input_data: NDArray,
    *,
    patch_shape: tuple[int, int, int, int],
    patch_overlap: tuple[int, int, int, int],
    mask_threshold: float,
    recombination: str | Recombination,
    method: str,
    mask: NDArray | None,
    noise_std: NDArray | float | None = None,
    batch_size: int = 0,
    extra_output: ExtraOutput = _NO_EXTRA_OUTPUT,
    **kwargs,
):
    """Denoise loop for the gpu version of patch-denoise.

    Parameters
    ----------
    $standard_config

    $noise_std

    extra_output: ExtraOutput, optional
        Bitmask of which optional outputs to return.

    Returns
    -------
    tuple
        ``(denoised, weights, var_map, rank_map, counts)``; any entry not
        requested via ``extra_output`` is ``None``.
    """
    pad = None
    if recombination == Recombination.CENTER:
        # Every voxel must be the center of some patch: pad by reflection.
        patch_shape = _patch_param(patch_shape, input_data.shape)
        patch_overlap = _patch_param(patch_overlap, input_data.shape)
        pad = center_padding(patch_shape, input_data.shape, mask)
        if any(lo or hi for lo, hi in pad):
            log.info(f"Padding {pad} for 'center' recombination.")
            input_data = np.pad(input_data, pad, mode="reflect")
            if mask is not None:
                mask = np.pad(mask, pad[: mask.ndim], mode="reflect")
            if isinstance(noise_std, np.ndarray):
                noise_std = np.pad(noise_std, pad[: noise_std.ndim], mode="reflect")
            padded_shape = input_data.shape
        else:
            pad = None

    # ensure single-precision for GPU compute.
    if np.iscomplexobj(input_data):
        input_data = input_data.astype(np.complex64, copy=False)
        compute_dtype = torch.complex64
    else:
        input_data = input_data.astype(np.float32, copy=False)
        compute_dtype = torch.float32

    squeeze_z = input_data.ndim == 3
    if squeeze_z:  # 2D + T
        data_shape = input_data.shape
        input_data = input_data[:, :, None, :]
        patch_shape = (patch_shape[0], patch_shape[1], 1, patch_shape[2])
        patch_overlap = (patch_overlap[0], patch_overlap[1], 0, patch_overlap[2])
        if mask is not None:
            if mask.shape == data_shape:
                mask = mask[:, :, None, :]
            elif mask.shape == data_shape[:-1]:
                mask = mask[:, :, None]
        if isinstance(noise_std, np.ndarray):
            noise_arr: NDArray = noise_std
            if noise_arr.shape == data_shape:
                noise_std = noise_arr[:, :, None, :]
            elif noise_arr.shape == data_shape[:-1]:
                noise_std = noise_arr[:, :, None]

    if recombination == Recombination.CENTER:
        check_center_recombination_overlap(patch_shape, patch_overlap, input_data.shape)
        # The synthetic Z axis injected above for 2D+T inputs always has
        # patch_shape == data_shape == 1; it isn't a real spatial axis and
        # must be excluded from the full-extent check below.
        spatial_patch = patch_shape[:2] if squeeze_z else patch_shape[:-1]
        spatial_data = input_data.shape[:2] if squeeze_z else input_data.shape[:-1]
        if any(ps == ds for ps, ds in zip(spatial_patch, spatial_data)):
            raise NotImplementedError(
                "GPU 'center' recombination only supports a full-extent "
                "(patch_shape == data_shape) axis on the last (time) axis; "
                "spatial axes must be smaller than the data shape. "
                "Use the CPU backend for this configuration."
            )

    # Time axis spans the whole data extent (e.g. from a "-1" patch/overlap):
    # "center" recombination then keeps the whole time profile at the
    # spatial center instead of collapsing it to a single time point too.
    full_time = patch_shape[-1] == input_data.shape[-1]

    # Create the Dataset
    if batch_size == 0:
        batch_size = autotune_batch_size(
            method,
            patch_shape,
            recombination,
            dtype=compute_dtype,
            full_time=full_time,
            **kwargs,
        )

    # Move the full volume to GPU once and gather patches directly from it:
    input_data_ = torch.from_numpy(input_data).cuda().contiguous()
    patch_dataset = PatchDataset(
        input_data_,
        patch_shape=patch_shape,
        patch_overlap=patch_overlap,
        mask=mask,
        noise_map=noise_std,
        mask_threshold=mask_threshold,
    )

    # Setup the denoiser model on GPU
    denoiser = make_denoiser(
        method,
        patch_shape=patch_shape,
        recombination=recombination,
        batch_size=batch_size,
        full_time=full_time,
        dtype=compute_dtype,
        **kwargs,
    )

    log.info(f"Processing {len(patch_dataset)} patches with batch size {batch_size}...")

    # zeros_like keeps the exact strides of the input, which the fused path
    # (reading and accumulating in place, at the same offsets) relies on.
    out_weights = torch.zeros_like(input_data_, dtype=torch.float32)
    out_acc = torch.zeros_like(input_data_)

    out_var_map: torch.Tensor | None = None
    out_rank_map: torch.Tensor | None = None
    out_counts: torch.Tensor | None = None
    ones_buf: torch.Tensor | None = None

    if extra_output & _NEEDS_COUNT:
        out_var_map = torch.zeros(input_data_.shape, dtype=torch.float32, device="cuda")
    if extra_output & ExtraOutput.RANK:
        out_rank_map = torch.zeros(input_data_.shape, dtype=torch.int32, device="cuda")
    if extra_output & _NEEDS_COUNT:
        out_counts = torch.zeros(input_data_.shape, dtype=torch.int32, device="cuda")
        ones_buf = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    var_apriori_by_patch = patch_dataset.var_apriori_by_patch
    use_var_apriori = var_apriori_by_patch is not None and "optimal" in method

    if recombination == Recombination.CENTER:
        # Absolute offset from a patch's top-left corner to its center
        # voxel; a patch's center is unique to it (distinct grid positions
        # give distinct centers), so writing it directly needs no atomics.
        center_offset = torch.tensor(
            [p // 2 for p in patch_shape], dtype=torch.int64, device="cuda"
        )

    # Read the patches in place instead of gathering a (B, *patch_shape) batch;
    # their weighted reconstruction is scattered from the same kernel.
    use_fused = _uses_fused(compute_dtype, patch_shape)

    n_patches = len(patch_dataset)
    for start in tqdm(range(0, n_patches, batch_size), unit_scale=batch_size):
        stop = min(start + batch_size, n_patches)
        denoiser_kwargs = {}
        if use_var_apriori:
            denoiser_kwargs["var_apriori"] = var_apriori_by_patch[start:stop]

        if recombination == Recombination.CENTER:
            if use_fused:
                gpu_bases, gpu_indices = patch_dataset.get_offsets(start, stop)
                gpu_center, gpu_weight, gpu_var_est, gpu_rank = denoiser.forward_center(
                    input_data_, gpu_bases, **denoiser_kwargs
                )
            else:
                gpu_in, gpu_indices = patch_dataset.get_batch(start, stop)
                gpu_center, gpu_weight, gpu_var_est, gpu_rank = denoiser(
                    gpu_in, **denoiser_kwargs
                )
            center_coords = gpu_indices + center_offset
            center_pos: tuple[slice, ...] = tuple(center_coords.unbind(-1))
            if full_time:
                center_pos = (*center_coords[:, :3].unbind(-1), slice(None))
            out_acc[center_pos] = gpu_center
            out_weights[center_pos] = gpu_weight
            # filter for last batch if it is smaller than batch_size
            batch_ones = ones_buf[: stop - start] if ones_buf is not None else None
            for out_map, value in zip(
                (out_var_map, out_rank_map, out_counts),
                (gpu_var_est, gpu_rank, batch_ones),
            ):
                if out_map is not None and value is not None:
                    if full_time:
                        value = value.unsqueeze(-1)
                    out_map[center_pos] = value
        elif use_fused:
            gpu_bases, gpu_indices = patch_dataset.get_offsets(start, stop)
            gpu_weight, gpu_var_est, gpu_rank = denoiser.forward_patches(
                input_data_, gpu_bases, out_acc, out_weights, **denoiser_kwargs
            )
            for out_map, value in zip(
                (out_var_map, out_rank_map, out_counts),
                (gpu_var_est, gpu_rank, ones_buf),
            ):
                if out_map is not None and value is not None:
                    launch_broadcast_add(out_map, value, gpu_indices, patch_shape)
        else:
            gpu_in, gpu_indices = patch_dataset.get_batch(start, stop)
            gpu_out, gpu_weight, gpu_var_est, gpu_rank = denoiser(
                gpu_in, **denoiser_kwargs
            )
            launch_accumulate_recon(
                out_acc, gpu_out, gpu_weight, gpu_indices, patch_shape
            )
            for out_map, value in zip(
                (out_weights, out_var_map, out_rank_map, out_counts),
                (gpu_weight, gpu_var_est, gpu_rank, ones_buf),
            ):
                if out_map is not None and value is not None:
                    launch_broadcast_add(out_map, value, gpu_indices, patch_shape)

    # free the dataset (every patch has been processed), but keep the mask.
    mask_arr = patch_dataset.mask.to(device="cuda", dtype=torch.bool)
    del patch_dataset, input_data_
    torch.cuda.empty_cache()

    zero_gpu = torch.tensor(0, device="cuda", dtype=torch.float32)
    zero_gpu_cpx = torch.tensor(0, device="cuda", dtype=out_acc.dtype)
    out_weights_safe = torch.where(
        out_weights == 0, torch.ones_like(out_weights), out_weights
    )
    out_acc /= out_weights_safe
    out_acc[~mask_arr] = zero_gpu_cpx
    out_acc = out_acc.cpu().numpy()

    if out_counts is not None:
        out_counts_safe = torch.where(
            out_counts == 0, torch.ones_like(out_counts), out_counts
        )

    if ExtraOutput.NOISE_STD in extra_output:
        assert out_var_map is not None
        out_var_map /= out_counts_safe
        out_var_map = torch.sqrt(out_var_map)
        out_var_map[~mask_arr] = zero_gpu
        out_var_map = out_var_map.cpu().numpy()
    else:
        out_var_map = None

    if ExtraOutput.RANK in extra_output:
        assert out_rank_map is not None
        out_rank_map = out_rank_map.to(dtype=torch.float32)
        out_rank_map /= out_counts_safe
        out_rank_map[~mask_arr] = zero_gpu
        out_rank_map = out_rank_map.cpu().numpy()
    else:
        out_rank_map = None

    if ExtraOutput.COUNT in extra_output:
        assert out_counts is not None
        out_counts[~mask_arr] = 0
        out_counts = out_counts.cpu().numpy()
    else:
        out_counts = None

    if ExtraOutput.WEIGHTS in extra_output:
        out_weights[~mask_arr] = zero_gpu
        out_weights = out_weights.cpu().numpy()
    else:
        out_weights = None

    if squeeze_z:
        out_acc, out_weights, out_var_map, out_rank_map, out_counts = (
            arr.squeeze(-2) if arr is not None else None
            for arr in (out_acc, out_weights, out_var_map, out_rank_map, out_counts)
        )
    return out_acc, out_weights, out_var_map, out_rank_map, out_counts
