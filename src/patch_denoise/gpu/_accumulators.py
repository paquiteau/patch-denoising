"""Triton atomic accumulators to recombine the patches to the main array."""

import torch
import triton
from triton import language as tl


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
