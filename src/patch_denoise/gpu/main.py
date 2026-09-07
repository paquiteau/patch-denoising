"""Main loop for the gpu version of patch-denoise."""

import logging

import numpy as np
import torch
import triton
import triton.language as tl
from numpy.typing import NDArray
from tqdm.rich import tqdm

from ..space_time.base import (
    ExtraOutput,
    Recombination,
    check_center_recombination_overlap,
)
from .._docs import fill_doc
from .autotune import autotune_batch_size
from .dataloader import PatchDataset
from .denoiser import MPPCADenoiser, OptimalSVDDenoiser

log = logging.getLogger(__name__)

_NEEDS_COUNT = ExtraOutput.COUNT | ExtraOutput.NOISE_STD | ExtraOutput.RANK
_NO_EXTRA_OUTPUT = ExtraOutput(0)


@triton.jit
def __accumulate_recon_kernel(
    global_out_ptr,  # Interpreted as float*
    recon_ptr,  # Interpreted as float*
    weights_ptr,  # per-patch scalar weight, float*
    coords_ptr,
    stride_out0: tl.constexpr,
    stride_out1: tl.constexpr,
    stride_out2: tl.constexpr,
    stride_out3: tl.constexpr,
    stride_rep_b: tl.constexpr,
    stride_rep_h: tl.constexpr,
    stride_rep_w: tl.constexpr,
    stride_rep_d: tl.constexpr,
    stride_rep_t: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    PD: tl.constexpr,
    PT: tl.constexpr,
    BATCH_SIZE,
    BLOCK_SIZE: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
):
    """Atomic-accumulate a batch of weighted reconstructed patches.

    Process one batch element per launch, with each thread block handling one pixel.
    """
    pid = tl.program_id(0)
    if pid >= BATCH_SIZE:
        return

    off_i = tl.load(coords_ptr + pid * 4 + 0)
    off_j = tl.load(coords_ptr + pid * 4 + 1)
    off_k = tl.load(coords_ptr + pid * 4 + 2)
    off_l = tl.load(coords_ptr + pid * 4 + 3)
    w = tl.load(weights_ptr + pid)

    for p_idx in range(0, PH * PW * PD * PT, BLOCK_SIZE):
        offsets = p_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (PH * PW * PD * PT)

        # 4D index logic
        curr_l = offsets % PT
        curr_k = (offsets // PT) % PD
        curr_j = (offsets // (PT * PD)) % PW
        curr_i = offsets // (PT * PD * PW)

        # Logical element index
        g_idx = (
            (off_i + curr_i) * stride_out0
            + (off_j + curr_j) * stride_out1
            + (off_k + curr_k) * stride_out2
            + (off_l + curr_l) * stride_out3
        )

        r_idx = (
            pid * stride_rep_b
            + curr_i * stride_rep_h
            + curr_j * stride_rep_w
            + curr_k * stride_rep_d
            + curr_l * stride_rep_t
        )

        if IS_COMPLEX:
            # We are using float32 pointers, so we multiply logical index by 2
            # to hit the interleaved Real/Imag parts.
            r_ptr_real = recon_ptr + 2 * r_idx
            g_ptr_real = global_out_ptr + 2 * g_idx

            # Atomic Add Real
            val_r = tl.load(r_ptr_real, mask=mask)
            tl.atomic_add(g_ptr_real, val_r * w, mask=mask)

            # Atomic Add Imaginary (Next float32 over)
            val_i = tl.load(r_ptr_real + 1, mask=mask)
            tl.atomic_add(g_ptr_real + 1, val_i * w, mask=mask)
        else:
            tl.atomic_add(
                global_out_ptr + g_idx,
                tl.load(recon_ptr + r_idx, mask=mask) * w,
                mask=mask,
            )


@triton.jit
def __broadcast_add_kernel(
    global_ptr,
    scalar_ptr,  # one value per patch, broadcast-added over its whole footprint
    coords_ptr,
    stride_out0: tl.constexpr,
    stride_out1: tl.constexpr,
    stride_out2: tl.constexpr,
    stride_out3: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    PD: tl.constexpr,
    PT: tl.constexpr,
    BATCH_SIZE,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomic-add one scalar per patch across its whole footprint.

    The weights/var-map/rank-map/count accumulators all reduce to this same
    operation (a per-patch scalar -- weight, variance estimate, rank, or a
    constant 1 for counting -- broadcast-added over the patch's voxels), so
    they share this one kernel instead of duplicating the indexing logic
    per output.
    """
    pid = tl.program_id(0)
    if pid >= BATCH_SIZE:
        return

    off_i = tl.load(coords_ptr + pid * 4 + 0)
    off_j = tl.load(coords_ptr + pid * 4 + 1)
    off_k = tl.load(coords_ptr + pid * 4 + 2)
    off_l = tl.load(coords_ptr + pid * 4 + 3)
    s = tl.load(scalar_ptr + pid)

    for p_idx in range(0, PH * PW * PD * PT, BLOCK_SIZE):
        offsets = p_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (PH * PW * PD * PT)

        curr_l = offsets % PT
        curr_k = (offsets // PT) % PD
        curr_j = (offsets // (PT * PD)) % PW
        curr_i = offsets // (PT * PD * PW)

        g_idx = (
            (off_i + curr_i) * stride_out0
            + (off_j + curr_j) * stride_out1
            + (off_k + curr_k) * stride_out2
            + (off_l + curr_l) * stride_out3
        )

        tl.atomic_add(global_ptr + g_idx, s, mask=mask)


def launch_accumulate_recon(
    out_acc: torch.Tensor,
    recon: torch.Tensor,
    weights: torch.Tensor,
    coords: torch.Tensor,
    patch_shape: tuple[int, int, int, int],
):
    """Launch the Triton kernel accumulating the weighted reconstruction."""
    is_complex = recon.is_complex()

    # TRICK: Interpret the tensors as float32 regardless of actual type.
    # This ensures Triton's pointer arithmetic is in 4-byte increments.
    # .view(torch.float32) does not copy; it just re-interprets the pointer.
    if is_complex:
        p_out = out_acc.view(torch.float32)
        p_recon = recon.view(torch.float32)
    else:
        p_out = out_acc
        p_recon = recon

    grid = (recon.shape[0],)
    __accumulate_recon_kernel[grid](
        p_out,
        p_recon,
        weights,
        coords,
        *out_acc.stride(),  # type: ignore
        *recon.stride(),
        *patch_shape,  # type: ignore
        BATCH_SIZE=recon.shape[0],  # type: ignore
        BLOCK_SIZE=1024,  # type: ignore
        IS_COMPLEX=is_complex,  # type: ignore
    )


def launch_broadcast_add(
    target: torch.Tensor,
    scalar_per_patch: torch.Tensor,
    coords: torch.Tensor,
    patch_shape: tuple[int, int, int, int],
):
    """Atomic-add one scalar per patch across its footprint into ``target``.

    Shared launcher for the weights/var-map/rank-map/count accumulators.
    """
    grid = (coords.shape[0],)
    __broadcast_add_kernel[grid](
        target,
        scalar_per_patch,
        coords,
        *target.stride(),  # type: ignore
        *patch_shape,  # type: ignore
        BATCH_SIZE=coords.shape[0],  # type: ignore
        BLOCK_SIZE=1024,  # type: ignore
    )


def make_denoiser(
    method,
    patch_shape,
    recombination,
    batch_size,
    compile=True,
    full_time=False,
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

    if compile:
        logging.info("starting module compilation.")
        with torch.inference_mode():
            # warm up to create working memory
            dummy_input = torch.randn(
                batch_size, *patch_shape, device="cuda", dtype=torch.float32
            )
            denoiser(dummy_input)
            # Triton bmm templates routinely exceed this GPU's shared-memory
            # budget for our patch sizes (OutOfMemoryError during autotuning)
            # and never beat cuBLAS anyway, so skip them and go straight to
            # ATEN/cuBLAS for matmuls.
            torch._inductor.config.max_autotune_gemm_backends = "ATEN"
            denoiser = torch.compile(
                denoiser,
                fullgraph=True,
                # "max-autotune" crashes
                # see https://github.com/pytorch/pytorch/issues/195731
                mode="max-autotune-no-cudagraphs",
            )  # Compile the model for faster inference
            # warm up the model with a dummy input to trigger compilation before timing
            denoiser(dummy_input)
        logging.info("Model compiled and warmed up on GPU.")
    # Clear overhead memory from autotuning benchmarks
    torch.cuda.empty_cache()

    return denoiser  # type: ignore


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
    compile: bool = False,
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
    else:
        input_data = input_data.astype(np.float32, copy=False)

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

    # Create the Dataset
    if batch_size == 0:
        batch_size = autotune_batch_size(method, patch_shape, recombination, **kwargs)

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

    # Time axis spans the whole data extent (e.g. from a "-1" patch/overlap):
    # "center" recombination then keeps the whole time profile at the
    # spatial center instead of collapsing it to a single time point too.
    full_time = patch_shape[-1] == input_data.shape[-1]

    # Setup the denoiser model on GPU
    denoiser = make_denoiser(
        method,
        patch_shape=patch_shape,
        recombination=recombination,
        batch_size=batch_size,
        compile=compile,
        full_time=full_time,
        **kwargs,
    )

    log.info(f"Processing {len(patch_dataset)} patches with batch size {batch_size}...")

    out_weights = torch.zeros(input_data_.shape, dtype=torch.float32, device="cuda")
    out_acc = torch.zeros(input_data_.shape, dtype=input_data_.dtype, device="cuda")
    out_var_map = (
        torch.zeros(input_data_.shape, dtype=torch.float32, device="cuda")
        if ExtraOutput.NOISE_STD in extra_output
        else None
    )
    out_rank_map = (
        torch.zeros(input_data_.shape, dtype=torch.int32, device="cuda")
        if ExtraOutput.RANK in extra_output
        else None
    )
    out_counts = (
        torch.zeros(input_data_.shape, dtype=torch.int32, device="cuda")
        if extra_output & _NEEDS_COUNT
        else None
    )
    # Reused as the per-patch "scalar" source for count accumulation, since
    # counting a patch is just broadcast-adding 1 over its footprint like
    # any other extra output.
    ones_buf = (
        torch.ones(batch_size, dtype=torch.int32, device="cuda")
        if extra_output & _NEEDS_COUNT
        else None
    )

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
            if full_time:
                # gpu_center/gpu_weight carry a full time profile per patch;
                # index the spatial center only and write the whole time
                # axis, instead of also collapsing it to a single point.
                center_pos = (*center_coords[:, :3].unbind(-1), slice(None))
            else:
                center_pos = center_coords.unbind(-1)
            out_acc[center_pos] = gpu_center
            out_weights[center_pos] = gpu_weight
            for out_map, value in zip(
                (out_var_map, out_rank_map, out_counts),
                (gpu_var_est, gpu_rank, ones_buf),
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
    # Voxels never any patch's center/never covered (e.g. near boundaries)
    # have weight 0 and accumulated nothing: divide by 1 there instead of 0
    # to avoid 0/0 -> NaN (matches the CPU path's explicit zero-fill).
    out_weights_safe = torch.where(
        out_weights == 0, torch.ones_like(out_weights), out_weights
    )
    out_acc /= out_weights_safe
    out_acc[~mask_arr] = zero_gpu_cpx
    out_acc = out_acc.cpu().numpy()

    # var_map/rank_map normalize by out_counts below: do that before the
    # with_counts block zeroes it out for return.
    if ExtraOutput.NOISE_STD in extra_output:
        assert out_var_map is not None
        assert out_counts is not None
        out_var_map /= out_counts
        out_var_map = torch.sqrt(out_var_map)
        out_var_map[~mask_arr] = zero_gpu
        out_var_map = out_var_map.cpu().numpy()
    else:
        out_var_map = None

    if ExtraOutput.RANK in extra_output:
        assert out_rank_map is not None
        assert out_counts is not None
        out_rank_map = out_rank_map.to(dtype=torch.float32)
        out_rank_map[~mask_arr] = zero_gpu
        out_rank_map /= out_counts
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
