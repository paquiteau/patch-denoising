"""Main loop for the gpu version of patch-denoise."""

import logging

import numpy as np
import torch
import triton
import triton.language as tl
from numpy.typing import NDArray
from tqdm.rich import tqdm

from .autotune import autotune_batch_size
from .dataloader import PatchDataset
from .denoiser import MPPCADenoiser, OptimalSVDDenoiser

log = logging.getLogger(__name__)


@triton.jit
def __atomic_accumulate_kernel(
    global_out_ptr,  # Interpreted as float*
    global_weights_ptr,  # Interpreted as float*
    global_var_est_ptr,  # Interpreted as float*
    global_rank_ptr,  # Interpreted as float*
    global_count_ptr,  # Interpreted as uint32*
    recon_ptr,  # Interpreted as float*
    weights_ptr,  # Interpreted as float*
    varest_ptr,  # Interpreted as float*
    rank_ptr,  # Interpreted as float*
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
    """Atomic accumulation kernel in Triton.

    Process one batch element per launch, with each thread block handling one pixel
    """
    pid = tl.program_id(0)
    if pid >= BATCH_SIZE:
        return

    # 1. Load coordinates and weight (Scalars)
    off_i = tl.load(coords_ptr + pid * 4 + 0)
    off_j = tl.load(coords_ptr + pid * 4 + 1)
    off_k = tl.load(coords_ptr + pid * 4 + 2)
    off_l = tl.load(coords_ptr + pid * 4 + 3)
    w = tl.load(weights_ptr + pid)
    v = tl.load(varest_ptr + pid)
    r = tl.load(rank_ptr + pid)

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

        # Weights are always 1 float per logical pixel
        tl.atomic_add(global_var_est_ptr + g_idx, v, mask=mask)
        tl.atomic_add(global_rank_ptr + g_idx, r, mask=mask)
        tl.atomic_add(global_weights_ptr + g_idx, w, mask=mask)
        tl.atomic_add(global_count_ptr + g_idx, 1, mask=mask)


def launch_triton(
    out_acc: torch.Tensor,
    out_weights: torch.Tensor,
    out_var_map: torch.Tensor,
    out_rank_map: torch.Tensor,
    out_counts: torch.Tensor,
    recon: torch.Tensor,
    weights: torch.Tensor,
    var_est: torch.Tensor,
    rank_est: torch.Tensor,
    coords: torch.Tensor,
    patch_shape: tuple[int, int, int, int],
):
    """Launch the Triton kernel for atomic accumulation."""
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
    __atomic_accumulate_kernel[grid](
        p_out,
        out_weights,
        out_var_map,
        out_rank_map,
        out_counts,
        p_recon,
        weights,
        var_est,
        rank_est,
        coords,
        *out_acc.stride(),  # type: ignore
        *recon.stride(),
        *patch_shape,  # type: ignore
        BATCH_SIZE=recon.shape[0],  # type: ignore
        BLOCK_SIZE=1024,  # type: ignore
        IS_COMPLEX=is_complex,  # type: ignore
    )


def make_denoiser(
    method, patch_shape, recombination, batch_size, compile=True, **kwargs
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
            **kwargs,
        )
    elif method == "mp-pca":
        denoiser = MPPCADenoiser(
            patch_shape=patch_shape,
            recombination=recombination,
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
    batch_size: int | str = "auto",
    compile: bool = False,
    **kwargs,
):
    """Denoise loop for the gpu version of patch-denoise."""
    if recombination == "center":
        raise ValueError(
            "recombination='center' is not supported on the GPU backend "
            "(only 'weighted' and 'average'/'mean' are implemented); use "
            "the CPU backend for 'center'."
        )

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

    # Create the Dataset
    if batch_size == "auto":
        batch_size = autotune_batch_size(method, patch_shape, recombination, **kwargs)
    else:
        batch_size = int(batch_size)
    # setup dataset and dataloader using pytorch api:
    input_data_ = torch.from_numpy(input_data)
    patch_dataset = PatchDataset(
        input_data_,
        patch_shape=patch_shape,
        patch_overlap=patch_overlap,
        mask=mask,
        noise_map=noise_std,
        mask_threshold=mask_threshold,
    )
    loader = torch.utils.data.DataLoader(
        patch_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,  # Pin memory for faster CPU-GPU transfer
    )

    # Setup the denoiser model on GPU
    denoiser = make_denoiser(
        method,
        patch_shape=patch_shape,
        recombination=recombination,
        batch_size=batch_size,
        compile=compile,
        **kwargs,
    )

    log.info(f"Processing {len(patch_dataset)} patches with batch size {batch_size}...")

    out_weights = torch.zeros(input_data_.shape, dtype=torch.float32, device="cuda")
    out_var_map = torch.zeros(input_data_.shape, dtype=torch.float32, device="cuda")
    out_rank_map = torch.zeros(input_data_.shape, dtype=torch.uint32, device="cuda")
    out_counts = torch.zeros(input_data_.shape, dtype=torch.uint32, device="cuda")
    out_acc = torch.zeros(input_data_.shape, dtype=input_data_.dtype, device="cuda")

    var_apriori_by_patch = patch_dataset.var_apriori_by_patch
    use_var_apriori = var_apriori_by_patch is not None and "optimal" in method

    cursor = 0
    for patches, indices in tqdm(loader, unit_scale=batch_size):
        gpu_in = patches.cuda(non_blocking=True)
        gpu_indices = indices.cuda(non_blocking=True)
        if use_var_apriori:
            var_batch = var_apriori_by_patch[cursor : cursor + gpu_in.shape[0]]
            gpu_out, gpu_weight, gpu_var_est, gpu_rank = denoiser(
                gpu_in, var_apriori=var_batch.cuda(non_blocking=True)
            )
        else:
            gpu_out, gpu_weight, gpu_var_est, gpu_rank = denoiser(gpu_in)
        cursor += gpu_in.shape[0]
        launch_triton(
            out_acc,
            out_weights,
            out_var_map,
            out_rank_map,
            out_counts,
            gpu_out,
            gpu_weight,
            gpu_var_est,
            gpu_rank,
            gpu_indices,
            patch_shape,
        )

    # Voxels outside the mask (or missed by every selected patch) never
    # accumulate any weight/count, so these divisions leave nan/inf there;
    # null them out via the mask instead of clamping the divisor.
    mask_arr = patch_dataset.mask.to(device="cuda", dtype=torch.bool)

    out_acc /= out_weights
    out_acc[~mask_arr] = 0
    out_acc = out_acc.cpu().numpy()

    out_var_map /= out_counts
    out_var_map = torch.sqrt(out_var_map)
    out_var_map[~mask_arr] = 0
    out_var_map = out_var_map.cpu().numpy()

    out_rank_map /= out_counts
    out_rank_map[~mask_arr] = 0
    out_rank_map = out_rank_map.cpu().numpy()

    out_weights[~mask_arr] = 0
    out_weights = out_weights.cpu().numpy()

    if squeeze_z:
        out_acc = out_acc[:, :, 0, :]
        out_weights = out_weights[:, :, 0, :]
        out_var_map = out_var_map[:, :, 0, :]
        out_rank_map = out_rank_map[:, :, 0, :]

    return out_acc, out_weights, out_var_map, out_rank_map
