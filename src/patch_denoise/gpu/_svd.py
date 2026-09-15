"""Fast batched SVD/eigensolve for patch denoising, replacing ``gesvda``.

``torch.linalg.svd(xc, driver="gesvda")`` silently promotes float32/complex64
input to FP64 internally inside cuSOLVER.

To circumvent this, we implement the same strategy as ``gesvda``, A reduction to
the Gram Matrix, followed by eigendecomposition (``cusolverDnXsyevBatched``),
with some twists:

- ``U`` is never computed. The only place ``U`` is used is
  ``U @ diag(s_shrink) @ Vh``, and ``U = Xc @ V / S``, so this collapses
  algebraically to a single ``(T, T)`` filter matrix
  ``M = V @ diag(s_shrink / S) @ Vh`` and one GEMM, ``x_denoised = Xc @ M + m``,
  instead of forming ``U`` (one GEMM) and then reconstructing (a second GEMM).
- We apply the centering of the Gram Matrix inline,
  to avoid materializing the centered tensor (``Xc = X - m``) in HBM.
"""

import logging

import nvmath.bindings.cublas as cublas
import nvmath.bindings.cusolver as cusolver
import nvmath.bindings.cusolverDn as cusolverDn
import torch
import triton
import triton.language as tl

log = logging.getLogger(__name__)

_CUDA_DTYPE = {torch.float32: 0, torch.complex64: 4}  # CUDA_R_32F, CUDA_C_32F


class _XsyevBatched:
    """Batched FP32/complex64 Hermitian eigensolver for fixed (batch_size, T).

    Wraps ``cusolverDnXsyevBatched`` with an explicit ``computeType`` (see
    module docstring). Workspace is sized once at construction for a fixed
    ``(batch_size, T)`` -- callers needing a different batch size (e.g. a
    smaller final batch) must construct another instance, see
    :class:`FastPatchSVD`.
    """

    def __init__(self, batch_size, T, dtype=torch.float32, device="cuda"):
        self.batch_size, self.T, self.dtype = batch_size, T, dtype
        self._cuda_dtype = _CUDA_DTYPE[dtype]

        self.handle = cusolverDn.create()
        cusolverDn.set_stream(self.handle, torch.cuda.current_stream().cuda_stream)
        self.params = cusolverDn.create_params()

        w_probe = torch.zeros(batch_size, T, device=device, dtype=torch.float32)
        g_probe = torch.zeros(batch_size, T, T, device=device, dtype=dtype)
        # create workspace buffers on device and host.
        self.dws, self.hws = cusolverDn.xsyev_batched_buffer_size(
            self.handle,
            self.params,
            cusolver.EigMode.VECTOR,
            cublas.FillMode.LOWER,
            T,
            self._cuda_dtype,
            g_probe.data_ptr(),
            T,
            0,
            w_probe.data_ptr(),  # CUDA_R_32F for eigenvalues (always real)
            self._cuda_dtype,
            batch_size,
        )
        self.d_buf = self.h_buf = None
        if self.dws:
            self.d_buf = torch.empty(self.dws, device=device, dtype=torch.uint8)
        if self.hws:
            self.h_buf = torch.empty(self.hws, dtype=torch.uint8)

        self.info = torch.zeros(batch_size, device=device, dtype=torch.int32)

    def __del__(self):
        try:
            cusolverDn.destroy_params(self.params)
            cusolverDn.destroy(self.handle)
        except Exception:
            log.debug("Failed to release cuSOLVER handle/params.", exc_info=True)

    def __call__(self, g):
        """
        Compute the batched eigendecomposition of a (B,T,T) Hermitian/symmetric matrix.

        Parameters
        ----------
        g : (B,T,T) tensor, float32 or complex64

        Returns
        -------
        w : (B,T) tensor, float32
            Ascending real eigenvalues. The eigenvectors are returned in-place in ``g``.

        Notes
        -----
        The eigenvectors are returned in-place in ``g``. The eigenvalues are always
        real.
        """
        B, T = self.batch_size, self.T
        cusolverDn.set_stream(self.handle, torch.cuda.current_stream().cuda_stream)
        w = torch.empty(B, T, device=g.device, dtype=torch.float32)
        cusolverDn.xsyev_batched(
            self.handle,
            self.params,
            cusolver.EigMode.VECTOR,
            cublas.FillMode.LOWER,
            T,
            self._cuda_dtype,
            g.data_ptr(),
            T,
            0,
            w.data_ptr(),
            self._cuda_dtype,
            self.d_buf.data_ptr() if self.d_buf is not None else 0,
            self.dws,
            self.h_buf.data_ptr() if self.h_buf is not None else 0,
            self.hws,
            self.info.data_ptr(),
            B,
        )
        if torch.any(self.info != 0):
            n_bad = int(torch.count_nonzero(self.info).item())
            log.warning(
                f"cuSOLVER xsyevBatched failed to converge on {n_bad}/{B} "
                "patch(es) in this batch; their eigenvectors/values are invalid."
            )
        return w


# Fixed kernel launch parameters for the fused kernels below.
_GRAM_REAL_CFG = dict(BLOCK_N=64, num_warps=4, num_stages=3)
_RECON_REAL_CFG = dict(BLOCK_N=64, num_warps=8, num_stages=3)

# Both fused kernels hold a (BLOCK_T, BLOCK_T) accumulator/filter tile in
# shared memory, at the above (BLOCK_N, num_stages) pipelining depth; that
# blows past typical GPU shared-memory limits once BLOCK_T reaches 256
# (empirically: T <= 128 launches fine, T in (128, 256] hits
# ``triton.runtime.errors.OutOfResources`` -- e.g. a full-time-extent patch
# on real fMRI data, T ~ 300). Eigh falls back to the honest (unfused) path
# above this cap -- see ``eigh``.
_FUSED_MAX_BLOCK_T = 128


def _block_t(T: int) -> int:
    """Return the padded T-tile size for the kernels below.

    ``tl.dot`` requires its contraction dimension to be >= 16 (a tensor-core
    hardware minimum) -- ``triton.next_power_of_2(T)`` alone isn't enough
    whenever T is itself an exact power of 2 below 16, since it then returns
    T unchanged.
    """
    return max(triton.next_power_of_2(T), 16)


@triton.jit
def _fused_centered_gram_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    N,
    T,
    stride_xb,
    stride_xn,
    stride_xt,
    stride_mb,
    stride_mt,
    stride_ob,
    stride_ot1,
    stride_ot2,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Compute the centered Gram matrix and mean of a batch of patches.

    ``out[b] = Xc[b]^T Xc[b]``, ``Xc[b] = x[b] - mean[b]``; ``mean_ptr`` is
    written too. Fused to avoid materializing the centered tensor in HBM; the
    mean is computed in-kernel (a first pass over N) rather than taken as an
    input, so there's no separate host-side reduction to wire up.
    """
    pid_b = tl.program_id(0)
    t_idx = tl.arange(0, BLOCK_T)
    t_mask = t_idx < T

    acc_sum = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        n_mask = n_idx < N
        mask2d = n_mask[:, None] & t_mask[None, :]
        x_ptrs = (
            x_ptr
            + pid_b * stride_xb
            + n_idx[:, None] * stride_xn
            + t_idx[None, :] * stride_xt
        )
        x_tile = tl.load(x_ptrs, mask=mask2d, other=0.0)
        acc_sum += tl.sum(x_tile, axis=0)
    m = acc_sum / N

    mean_ptrs = mean_ptr + pid_b * stride_mb + t_idx * stride_mt
    tl.store(mean_ptrs, m, mask=t_mask)

    acc = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        n_mask = n_idx < N
        mask2d = n_mask[:, None] & t_mask[None, :]
        x_ptrs = (
            x_ptr
            + pid_b * stride_xb
            + n_idx[:, None] * stride_xn
            + t_idx[None, :] * stride_xt
        )
        x_tile = tl.load(x_ptrs, mask=mask2d, other=0.0)
        xc_tile = tl.where(mask2d, x_tile - m[None, :], 0.0)
        acc += tl.dot(tl.trans(xc_tile), xc_tile, allow_tf32=False)

    out_mask = t_mask[:, None] & t_mask[None, :]
    out_ptrs = (
        out_ptr
        + pid_b * stride_ob
        + t_idx[:, None] * stride_ot1
        + t_idx[None, :] * stride_ot2
    )
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _fused_reconstruction_kernel(
    x_ptr,
    mean_ptr,
    mm_ptr,
    out_ptr,
    N,
    T,
    stride_xb,
    stride_xn,
    stride_xt,
    stride_mb,
    stride_mt,
    stride_mmb,
    stride_mmt1,
    stride_mmt2,
    stride_ob,
    stride_on,
    stride_ot,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Compute the reconstruction of a batch of patches.

    As for the fused Gram kernel, the centered tensor is never materialized in HBM.
    The mean is taken as an input, so we compute:
    out[b] = (x[b]-mean[b]) @ Mm[b] + mean[b].

    Where Mm[b] is the filter matrix computed from the SVD of the centered Gram matrix.

    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_idx < N
    t_idx = tl.arange(0, BLOCK_T)
    t_mask = t_idx < T
    mask2d = n_mask[:, None] & t_mask[None, :]

    mean_ptrs = mean_ptr + pid_b * stride_mb + t_idx * stride_mt
    m = tl.load(mean_ptrs, mask=t_mask, other=0.0)
    x_ptrs = (
        x_ptr
        + pid_b * stride_xb
        + n_idx[:, None] * stride_xn
        + t_idx[None, :] * stride_xt
    )
    x_tile = tl.load(x_ptrs, mask=mask2d, other=0.0)
    xc_tile = tl.where(mask2d, x_tile - m[None, :], 0.0)

    mm_mask = t_mask[:, None] & t_mask[None, :]
    mm_ptrs = (
        mm_ptr
        + pid_b * stride_mmb
        + t_idx[:, None] * stride_mmt1
        + t_idx[None, :] * stride_mmt2
    )
    Mm = tl.load(mm_ptrs, mask=mm_mask, other=0.0)
    out_tile = tl.dot(xc_tile, Mm, allow_tf32=False) + m[None, :]
    out_ptrs = (
        out_ptr
        + pid_b * stride_ob
        + n_idx[:, None] * stride_on
        + t_idx[None, :] * stride_ot
    )
    tl.store(out_ptrs, out_tile, mask=mask2d)


def _fused_centered_gram(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the batched centered Gram matrix and mean of a batch of patches.

    Parameters
    ----------
    x : (B, N, T) tensor, float32

    Returns
    -------
    gc : (B, T, T) tensor, float32
        Centered Gram matrix: ``gc[b] = (x[b]-mean[b])^T @ (x[b]-mean[b])``.
    m : (B, 1, T) tensor, float32
        Per-row mean: ``m[b] = mean(x[b], dim=0, keepdim=True)``.

    """
    B, N, T = x.shape
    m = torch.empty(B, T, device=x.device, dtype=torch.float32)
    gc = torch.empty(B, T, T, device=x.device, dtype=torch.float32)
    BLOCK_T = _block_t(T)
    _fused_centered_gram_kernel[(B,)](
        x,
        m,
        gc,
        N,
        T,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        m.stride(0),
        m.stride(1),
        gc.stride(0),
        gc.stride(1),
        gc.stride(2),
        BLOCK_T=BLOCK_T,  # type: ignore
        **_GRAM_REAL_CFG,  # type: ignore
    )
    return gc, m.unsqueeze(1)


def _fused_reconstruction(x, m, Mm):
    """Compute ``(x-m) @ Mm + m`` via the fused Triton kernel above. Float32 only.

    Parameters
    ----------
    x : (B, N, T) tensor, float32
        Input patches.
    m : (B, 1, T) tensor, float32
        Per-row mean.
    Mm : (B, T, T) tensor, float32
        Filter matrix computed from the SVD of the centered Gram matrix.

    Returns
    -------
    out : (B, N, T) tensor, float32
        Reconstructed patches: ``out[b] = (x[b]-m[b]) @ Mm[b] + m[b]``.
    """
    B, N, T = x.shape
    m2 = m.reshape(B, T)
    out = torch.empty_like(x)
    BLOCK_T = _block_t(T)
    grid = (B, triton.cdiv(N, _RECON_REAL_CFG["BLOCK_N"]))
    _fused_reconstruction_kernel[grid](
        x,
        m2,
        Mm,
        out,
        N,
        T,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        m2.stride(0),
        m2.stride(1),
        Mm.stride(0),
        Mm.stride(1),
        Mm.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_T=BLOCK_T,  # type: ignore
        **_RECON_REAL_CFG,  # type: ignore
    )
    return out


class FastPatchSVD(torch.nn.Module):
    """
    Fast batched SVD/eigensolve for patch denoising, replacing ``gesvda``.

    This class implements a custom SVD driver for a batch of tall-skinny
    matrices (patches) that avoids the FP64 promotion of ``gesvda`` in cuSOLVER.

    It computes the eigendecomposition of the centered Gram matrix and
    reconstructs the denoised patches without explicitly forming the left
    singular vectors.

    Moreover, it does not materialize the centered tensor nor the "U" matrix,
    which saves memory and computation time.

    Example
    -------
    >>> svd = FastPatchSVD(T=128)
    >>> s, vh, m, xc = svd.eigh(x_flat)
    >>> ratio = s / s.sum(dim=-1, keepdim=True) Shrink the singular values
    >>> x_denoised = svd.reconstruct(x_flat, m, xc, vh, ratio)
    """

    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self._solvers: dict[tuple[int, torch.dtype, torch.device], _XsyevBatched] = {}

    def _get_solver(self, B: int, dtype: torch.dtype, device) -> _XsyevBatched:
        key = (B, dtype, torch.device(device))
        solver = self._solvers.get(key)
        if solver is None:
            solver = _XsyevBatched(B, self.T, dtype=dtype, device=device)
            self._solvers[key] = solver
        return solver

    def eigh(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Eigendecompose the centered Gram matrix of a batch of flattened patches.

        Parameters
        ----------
        x_flat : (B, N, T) tensor, float32 or complex64.

        Returns
        -------
        s : (B, T) tensor
            Descending real singular values.
        vh : (B, T, T) tensor
            Right singular vectors, ``Vh`` row-convention (matches
            ``torch.linalg.svd``'s ``vh``). ``U`` is never computed.
        m : (B, 1, T) tensor
            Per-row mean.
        xc : (B, N, T) tensor or None
            Centered data: ``None`` for float32 (the fused Triton path never
            materializes it), the real centered tensor for complex64 (computed
            once here, reused by :meth:`reconstruct` so it isn't recomputed).
        """
        dtype = x_flat.dtype
        if dtype not in (torch.float32, torch.complex64):
            raise TypeError(
                f"FastPatchSVD only supports float32/complex64, got {dtype}"
            )

        if dtype == torch.float32 and _block_t(x_flat.shape[-1]) <= _FUSED_MAX_BLOCK_T:
            gc, m = _fused_centered_gram(x_flat)
            xc = None
        else:
            m = torch.mean(x_flat, dim=-2, keepdim=True)
            xc = x_flat - m
            xch = xc.conj().transpose(-2, -1) if dtype == torch.complex64 else xc.mT
            gc = torch.matmul(xch, xc)

        solver = self._get_solver(x_flat.shape[0], dtype, x_flat.device)
        w = solver(gc)  # in-place: gc now holds eigenvectors; w ascending real.
        # Row j of the raw output already equals conj(eigenvector_j) -- i.e.
        # exactly Vh's row j (see module docstring). Reverse for descending.
        vh = gc.flip(-2)
        s = w.clamp_min(0).sqrt().flip(-1)
        return s, vh, m, xc

    @staticmethod
    def _filter_matrix(vh: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """``M = V @ diag(ratio) @ Vh``, replacing the need to ever compute ``U``."""
        v = vh.conj().transpose(-2, -1) if vh.is_complex() else vh.transpose(-2, -1)
        return torch.matmul(v * ratio.unsqueeze(-2), vh)

    def reconstruct(
        self,
        x_flat: torch.Tensor,
        m: torch.Tensor,
        xc: torch.Tensor | None,
        vh: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Full (B, N, T) reconstruction: ``Xc @ M + m``.

        ``xc=None`` (float32) -> fused Triton reconstruction (never
        materializes the centered tensor); otherwise (complex64) ->
        ``torch.baddbmm(m, xc, M)``.
        """
        M = self._filter_matrix(vh, ratio)
        if xc is None:
            return _fused_reconstruction(x_flat, m, M)
        return torch.baddbmm(m, xc, M)

    def center_reconstruct(
        self,
        x_flat: torch.Tensor,
        m: torch.Tensor,
        vh: torch.Tensor,
        ratio: torch.Tensor,
        spatial_idx: int,
        time_idx: int | None = None,
    ) -> torch.Tensor:
        """Perform ``center`` recombination: one row or one scalar/patch.

        Parameters
        ----------
        x_flat : (B, N, T) tensor
            Input patches.
        m : (B, 1, T) tensor
            Per-row mean.
        vh : (B, T, T) tensor
            Right singular vectors, ``Vh`` row-convention (matches
            ``torch.linalg.svd``'s ``vh``). ``U`` is never computed
        ratio : (B, T) tensor
            Singular value shrinkage ratio.
        spatial_idx : int
            Index of the spatial location to reconstruct (0 <= spatial_idx < N).
        time_idx : int or None, optional
            Index of the time point to reconstruct (0 <= time_idx < T). If None,
            reconstructs the entire row (``(B, T)``).

        Returns
        -------
        out : (B, T) tensor if time_idx is None, else (B,)
            Reconstructed row or scalar for the specified spatial and time indices.

        """
        M = self._filter_matrix(vh, ratio)
        xc_center = x_flat[:, spatial_idx, :] - m[:, 0, :]  # (B,T)
        if time_idx is None:
            return torch.matmul(xc_center.unsqueeze(1), M).squeeze(1) + m[:, 0, :]
        M_col = M[:, :, time_idx]  # (B,T)
        return torch.sum(xc_center * M_col, dim=-1) + m[:, 0, time_idx]
