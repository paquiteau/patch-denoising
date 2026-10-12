"""Triton atomic accumulator of per-patch scalars into a volume."""

import torch
import triton
from triton import language as tl


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

    The var-map/rank-map/count accumulators all reduce to this same
    operation (a per-patch scalar -- variance estimate, rank, or a
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

        tl.atomic_add(global_ptr + g_idx, s, mask=mask, sem="relaxed")


def launch_broadcast_add(
    target: torch.Tensor,
    scalar_per_patch: torch.Tensor,
    coords: torch.Tensor,
    patch_shape: tuple[int, int, int, int],
):
    """Atomic-add one scalar per patch across its footprint into ``target``.

    Shared launcher for the var-map/rank-map/count accumulators.
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
