"""Main loop for the gpu version of patch-denoise."""

import logging
import os

import torch
import triton
import triton.language as tl
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .dataloader import PatchDataset
from .denoiser import MPPCADenoiser, OptimalSVDDenoiser


@triton.jit
def __atomic_accumulate_kernel(
    global_out_ptr,  # Interpreted as float*
    global_weights_ptr,  # Interpreted as float*
    recon_ptr,  # Interpreted as float*
    weights_ptr,  # Interpreted as float*
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
        tl.atomic_add(global_weights_ptr + g_idx, w, mask=mask)


def launch_triton(out_acc, out_weights, recon, weights, coords, patch_shape):
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
        p_recon,
        weights,
        coords,
        *out_acc.stride(),
        *recon.stride(),
        *patch_shape,
        BATCH_SIZE=recon.shape[0],
        BLOCK_SIZE=1024,  # type: ignore
        IS_COMPLEX=is_complex,
    )


def make_denoiser(
    args, noise_map, batch_size, compile=True, **kwargs
) -> torch.nn.Module:
    """Create a denoiser model on GPU."""
    print(args)
    if "optimal" in args.method:
        denoiser = OptimalSVDDenoiser(
            patch_shape=args.patch_shape,
            recombination=args.recombination,
            loss=args.method.split("-")[-1],
            **kwargs,
        )
    elif args.method == "mppca":
        denoiser = MPPCADenoiser(
            patch_shape=args.patch_shape,
            recombination=args.recombination,
            **kwargs,
        )
    else:
        raise ValueError(f"method {args.method} is not supported on GPU. ")

    denoiser = denoiser.cuda()  # Move model to GPU

    # cache compilation
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./.torch_cache"
    os.environ["TORCHINDUCTOR_CACHE_ENABLED"] = "1"
    torch.set_float32_matmul_precision("high")

    if compile:
        logging.info("starting module compilation.")
        with torch.inference_mode():
            # warm up to create working memory
            dummy_input = torch.randn(
                batch_size, *args.patch_shape, device="cuda", dtype=torch.float32
            )
            denoiser(dummy_input)
            denoiser = torch.compile(
                denoiser,
                fullgraph=True,
                mode="max-autotune",
            )  # Compile the model for faster inference
            # warm up the model with a dummy input to trigger compilation before timing
            denoiser(dummy_input)
        logging.info("Model compiled and warmed up on GPU.")
    # Clear overhead memory from autotuning benchmarks
    torch.cuda.empty_cache()

    return denoiser


@torch.inference_mode()
def main_gpu(
    args,
    input_data: NDArray,
    mask: NDArray | None,
    noise_map: NDArray | None,
    batch_size: int = 32,
    compile: bool = False,
    **kwargs,
):
    """Denoise loop for the gpu version of patch-denoise."""
    # Create the Dataset
    batch_size = int(batch_size)
    # setup dataset and dataloader using pytorch api:
    input_data_ = torch.from_numpy(input_data)
    patch_dataset = PatchDataset(
        input_data_,
        patch_shape=args.patch_shape,
        patch_overlap=args.patch_overlap,
        mask=mask,
        noise_map=noise_map,
        mask_threshold=args.mask_threshold,
    )
    loader = torch.utils.data.DataLoader(
        patch_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,  # Pin memory for faster CPU-GPU transfer
    )

    # Setup the denoiser model on GPU
    denoiser = make_denoiser(
        args, noise_map, batch_size=batch_size, compile=compile, **kwargs
    )

    N_STREAMS = 2
    streams = [torch.cuda.Stream() for _ in range(N_STREAMS)]
    logging.info(
        f"Processing {len(patch_dataset)} patches with batch size {batch_size}..."
    )

    out_weights = torch.zeros(input_data_.shape, dtype=input_data_.dtype, device="cuda")
    out_acc = torch.zeros(input_data_.shape, dtype=torch.float32, device="cuda")
    for i, (patches, indices) in enumerate(tqdm(loader, unit_scale=batch_size)):
        slot = i % 2
        stream = streams[slot]
        with torch.cuda.stream(stream):
            # H2D -> Compute -> D2H (All Asynchronous)
            gpu_in = patches.cuda(non_blocking=True)
            gpu_indices = indices.cuda(non_blocking=True)
            gpu_out, gpu_weight = denoiser(gpu_in)
            launch_triton(
                out_acc,
                out_weights,
                gpu_out,
                gpu_weight,
                gpu_indices,
                args.patch_shape,
            )

    out_acc /= out_weights.clamp(min=1e-8)  # Avoid division by zero
    out_acc = out_acc.cpu().numpy()
    out_weights = out_weights.cpu().numpy()

    return out_acc, out_weights, None
