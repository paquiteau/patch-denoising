"""Main loop for the gpu version of patch-denoise."""

import logging

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.rich import tqdm

from .._docs import fill_doc
from ..space_time.base import (
    DenoiserName,
    ExtraOutput,
    Recombination,
    check_center_recombination_overlap,
)
from ._accumulators import launch_accumulate_recon, launch_broadcast_add
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

    torch.set_float32_matmul_precision("high")

    # Warm up: builds FastPatchSVD's cuSOLVER workspace and JIT-compiles its
    # Triton kernels for this batch size before the tracked loop starts, so
    # construction errors fail fast instead of surfacing on the first batch.
    with torch.inference_mode():
        dummy_input = torch.randn(batch_size, *patch_shape, device="cuda", dtype=dtype)
        denoiser(dummy_input)
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
    recombination: str,
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

    if recombination == Recombination.CENTER:
        check_center_recombination_overlap(patch_shape, patch_overlap, input_data.shape)
        if any(ps == ds for ps, ds in zip(patch_shape[:-1], input_data.shape[:-1])):
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
    input_data_ = torch.from_numpy(input_data).cuda()
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

    out_weights = torch.zeros(input_data_.shape, dtype=torch.float32, device="cuda")
    out_acc = torch.zeros(input_data_.shape, dtype=input_data_.dtype, device="cuda")

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

    if recombination == "center":
        # Absolute offset from a patch's top-left corner to its center
        # voxel; a patch's center is unique to it (distinct grid positions
        # give distinct centers), so writing it directly needs no atomics.
        center_offset = torch.tensor(
            [p // 2 for p in patch_shape], dtype=torch.int64, device="cuda"
        )

    n_patches = len(patch_dataset)
    for start in tqdm(range(0, n_patches, batch_size), unit_scale=batch_size):
        stop = min(start + batch_size, n_patches)
        gpu_in, gpu_indices = patch_dataset.get_batch(start, stop)
        denoiser_kwargs = {}
        if use_var_apriori:
            denoiser_kwargs["var_apriori"] = var_apriori_by_patch[start:stop]

        if recombination == "center":
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
        else:
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
