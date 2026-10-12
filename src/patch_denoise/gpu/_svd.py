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
- Patches are centered along time (``m`` is the temporal mean of each voxel),
  inline in the Triton kernels, to avoid materializing the centered tensor
  (``Xc = X - m``) in HBM. A tile holds whole rows, so the mean is a register
  reduction.
- Centering along time makes the constant time course an exact null direction
  of ``Xc``: only the ``k = min(N, T - 1)`` non-trivial components are returned.
  This also removes the mean image -- by far the dominant component of a real
  patch -- from the Gram matrix, which keeps it well conditioned enough for an
  FP32 eigensolver to resolve the noise eigenvalues.
"""

import logging
import math

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


def fused_supported(dtype: torch.dtype, T: int) -> bool:
    """Whether the fused Triton kernels handle patches of this dtype and length."""
    return dtype == torch.float32 and _block_t(T) <= _FUSED_MAX_BLOCK_T


@triton.jit
def _patch_offsets(base, n, sx, sy, sz, PY: tl.constexpr, PZ: tl.constexpr):
    """Element offset of voxel ``n`` (C order) of a ``(PX, PY, PZ)`` patch."""
    return base + (n // (PY * PZ)) * sx + (n // PZ % PY) * sy + (n % PZ) * sz


@triton.jit
def _fused_centered_gram_kernel(
    x_ptr,
    base_ptr,
    out_ptr,
    T,
    sx,
    sy,
    sz,
    st,
    N: tl.constexpr,
    PY: tl.constexpr,
    PZ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Compute the Gram matrix of a batch of patches centered along time.

    ``out[b] = Xc[b]^T Xc[b]``, ``Xc[b] = X[b] - mean_t(X[b])``, with ``X[b]``
    the ``(N, T)`` patch whose first element is at ``x_ptr + base[b]``: patches
    are read in place, never gathered. A tile spans whole rows, so each
    voxel's temporal mean is computed in registers, in the same single pass
    over N that accumulates the Gram matrix.
    """
    pid_b = tl.program_id(0)
    base = tl.load(base_ptr + pid_b)
    t_idx = tl.arange(0, BLOCK_T)
    t_mask = t_idx < T

    acc = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        mask2d = (n_idx < N)[:, None] & t_mask[None, :]
        rows = _patch_offsets(base, n_idx, sx, sy, sz, PY, PZ)
        x_tile = tl.load(
            x_ptr + rows[:, None] + t_idx[None, :] * st, mask=mask2d, other=0.0
        )
        m = tl.sum(x_tile, axis=1) / T  # temporal mean of each voxel
        xc_tile = tl.where(mask2d, x_tile - m[:, None], 0.0)
        acc = tl.dot(tl.trans(xc_tile), xc_tile, acc, input_precision="ieee")

    out_mask = t_mask[:, None] & t_mask[None, :]
    out_ptrs = out_ptr + pid_b * T * T + t_idx[:, None] * T + t_idx[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _fused_reconstruction_kernel(
    x_ptr,
    base_ptr,
    mm_ptr,
    w_ptr,
    out_ptr,
    wsum_ptr,
    T,
    sx,
    sy,
    sz,
    st,
    N: tl.constexpr,
    PY: tl.constexpr,
    PZ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SCATTER: tl.constexpr,
):
    """Reconstruct a batch of patches: ``(X[b] - m[b]) @ Mm[b] + m[b]``.

    Patches are addressed as in the Gram kernel, and the temporal mean ``m``
    is again computed in registers. ``Mm[b]`` is the filter matrix computed
    from the eigendecomposition of the centered Gram matrix.

    With ``SCATTER``, ``w[b]`` times the result is atomically added back in
    place into ``out`` (which shares ``x``'s layout), and ``w[b]`` into
    ``wsum``: the reconstructed batch never exists in memory. Otherwise it is
    stored to a contiguous ``(B, N, T)`` ``out``.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    t_idx = tl.arange(0, BLOCK_T)
    t_mask = t_idx < T
    mask2d = (n_idx < N)[:, None] & t_mask[None, :]

    rows = _patch_offsets(tl.load(base_ptr + pid_b), n_idx, sx, sy, sz, PY, PZ)
    offsets = rows[:, None] + t_idx[None, :] * st
    x_tile = tl.load(x_ptr + offsets, mask=mask2d, other=0.0)
    m = tl.sum(x_tile, axis=1) / T
    xc_tile = tl.where(mask2d, x_tile - m[:, None], 0.0)

    mm_mask = t_mask[:, None] & t_mask[None, :]
    mm_ptrs = mm_ptr + pid_b * T * T + t_idx[:, None] * T + t_idx[None, :]
    Mm = tl.load(mm_ptrs, mask=mm_mask, other=0.0)
    out_tile = tl.dot(xc_tile, Mm, input_precision="ieee") + m[:, None]
    if SCATTER:
        w = tl.load(w_ptr + pid_b)
        # A sum needs no ordering between the atomics: the default
        # (acq_rel) fences cost ~3x on this kernel.
        tl.atomic_add(out_ptr + offsets, w * out_tile, mask=mask2d, sem="relaxed")
        tl.atomic_add(
            wsum_ptr + offsets, w + tl.zeros_like(out_tile), mask=mask2d, sem="relaxed"
        )
    else:
        out_ptrs = out_ptr + (pid_b * N + n_idx[:, None]) * T + t_idx[None, :]
        tl.store(out_ptrs, out_tile, mask=mask2d)


def _patch_layout(x, patch_shape):
    """Kernel addressing arguments: ``(T, sx, sy, sz, st)`` and ``(N, PY, PZ)``.

    ``x`` is either a ``(B, N, T)`` batch (``patch_shape=None``, read as
    ``(N, 1, 1)`` patches) or a volume of which ``patch_shape`` patches are
    read in place.
    """
    if patch_shape is None:
        _, N, T = x.shape
        return (T, x.stride(1), 0, 0, x.stride(2)), dict(N=N, PY=1, PZ=1)
    px, py, pz, T = patch_shape
    return (T, *x.stride()), dict(N=px * py * pz, PY=py, PZ=pz)


def _batch_bases(x):
    """Offset of each patch of a ``(B, N, T)`` batch, for the kernels above."""
    return torch.arange(x.shape[0], device=x.device, dtype=torch.int64) * x.stride(0)


def _fused_centered_gram(x, bases, patch_shape=None):
    """Compute the batched Gram matrix of patches centered along time.

    Parameters
    ----------
    x : tensor, float32
        ``(B, N, T)`` batch of patches, or a volume ``(X, Y, Z, T')`` read in
        place, see ``patch_shape``.
    bases : (B,) int64 tensor
        Element offset in ``x`` of each patch's first voxel.
    patch_shape : (px, py, pz, T) tuple, optional
        Shape of the patches, when ``x`` is a volume.

    Returns
    -------
    gc : (B, T, T) tensor, float32
        ``gc[b] = (X[b] - m[b])^T @ (X[b] - m[b])``, with ``m[b]`` the
        ``(N, 1)`` temporal mean of each voxel.
    """
    (T, *strides), meta = _patch_layout(x, patch_shape)
    B = bases.shape[0]
    gc = torch.empty(B, T, T, device=x.device, dtype=torch.float32)
    _fused_centered_gram_kernel[(B,)](
        x,
        bases,
        gc,
        T,
        *strides,
        BLOCK_T=_block_t(T),  # type: ignore
        **meta,
        **_GRAM_REAL_CFG,
    )
    return gc


def _fused_reconstruction(
    x, bases, Mm, patch_shape=None, weight=None, out=None, wsum=None
):
    """Compute ``(X - m) @ Mm + m`` via the fused Triton kernel above. Float32 only.

    Parameters
    ----------
    x, bases, patch_shape :
        Patches to reconstruct, as in :func:`_fused_centered_gram`.
    Mm : (B, T, T) tensor, float32
        Filter matrix computed from the eigendecomposition of the Gram matrix.
    weight : (B,) tensor, float32, optional
        Per-patch weights. When given, the weighted reconstruction is
        atomically added in place into ``out`` (same shape and strides as
        ``x``), and the weights into ``wsum`` (idem).

    Returns
    -------
    out : tensor, float32
        Without ``weight``, the reconstructed ``(B, N, T)`` patches:
        ``out[b] = (X[b] - m[b]) @ Mm[b] + m[b]``. Otherwise the input ``out``.
    """
    (T, *strides), meta = _patch_layout(x, patch_shape)
    B = bases.shape[0]
    scatter = weight is not None
    if scatter:
        if out is None or wsum is None:
            raise ValueError("Scattering needs both an `out` and a `wsum` accumulator.")
        if out.stride() != x.stride() or wsum.stride() != x.stride():
            raise ValueError("`out` and `wsum` must share the layout of `x`.")
    else:
        out = torch.empty(B, meta["N"], T, device=x.device, dtype=torch.float32)
    grid = (B, triton.cdiv(meta["N"], _RECON_REAL_CFG["BLOCK_N"]))
    _fused_reconstruction_kernel[grid](
        x,
        bases,
        Mm.contiguous(),
        weight,
        out,
        wsum,
        T,
        *strides,
        BLOCK_T=_block_t(T),  # type: ignore
        SCATTER=scatter,  # type: ignore
        **meta,
        **_RECON_REAL_CFG,
    )
    return out


class FastPatchSVD(torch.nn.Module):
    """
    Fast batched SVD/eigensolve for patch denoising, replacing ``gesvda``.

    This class implements a custom SVD driver for a batch of tall-skinny
    matrices (patches) that avoids the FP64 promotion of ``gesvda`` in cuSOLVER.

    It computes the eigendecomposition of the Gram matrix of the patches,
    centered along time, and reconstructs the denoised patches without
    explicitly forming the left singular vectors.

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Eigendecompose the Gram matrix of a batch of patches centered along time.

        Parameters
        ----------
        x_flat : (B, N, T) tensor, float32 or complex64.

        Returns
        -------
        s : (B, k) tensor
            Descending real singular values, ``k = min(N, T - 1)``: centering
            along time makes the constant time course an exact null direction,
            which is dropped.
        vh : (B, k, T) tensor
            Right singular vectors, ``Vh`` row-convention (matches
            ``torch.linalg.svd``'s ``vh``). ``U`` is never computed.
        m : (B, N, 1) tensor or None
            Temporal mean of each voxel; ``None`` for float32 (the fused
            Triton kernels compute it in registers).
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

        if fused_supported(dtype, x_flat.shape[-1]):
            gc = _fused_centered_gram(x_flat, _batch_bases(x_flat))
            m = xc = None
        else:
            m = torch.mean(x_flat, dim=-1, keepdim=True)
            xc = x_flat - m
            xch = xc.conj().transpose(-2, -1) if dtype == torch.complex64 else xc.mT
            gc = torch.matmul(xch, xc)

        s, vh = self._solve(gc, x_flat.shape[-2])
        return s, vh, m, xc

    def eigh_patches(
        self, vol: torch.Tensor, bases: torch.Tensor, patch_shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eigendecompose, as :meth:`eigh`, patches read in place from ``vol``.

        Float32 only, and only when :func:`fused_supported`.

        Parameters
        ----------
        vol : (X, Y, Z, T') tensor, float32
        bases : (B,) int64 tensor
            Element offset in ``vol`` of the first voxel of each patch.
        patch_shape : (px, py, pz, T) tuple

        Returns
        -------
        s, vh :
            As for :meth:`eigh`.
        """
        gc = _fused_centered_gram(vol, bases, patch_shape)
        return self._solve(gc, math.prod(patch_shape[:-1]))

    def _solve(self, gc: torch.Tensor, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        solver = self._get_solver(gc.shape[0], gc.dtype, gc.device)
        w = solver(gc)  # in-place: gc now holds eigenvectors; w ascending real.
        # Row j of the raw output already equals conj(eigenvector_j) -- i.e.
        # exactly Vh's row j (see module docstring). Reverse for descending,
        # and keep the k non-trivial components.
        k = min(N, self.T - 1)
        vh = gc.flip(-2)[:, :k]
        s = w.clamp_min(0).sqrt().flip(-1)[:, :k]
        return s, vh

    @staticmethod
    def _filter_matrix(vh: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """``M = V @ diag(ratio) @ Vh``, replacing the need to ever compute ``U``."""
        v = vh.conj().transpose(-2, -1) if vh.is_complex() else vh.transpose(-2, -1)
        return torch.matmul(v * ratio.unsqueeze(-2), vh)

    def reconstruct(
        self,
        x_flat: torch.Tensor,
        m: torch.Tensor | None,
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
            return _fused_reconstruction(x_flat, _batch_bases(x_flat), M)
        assert m is not None
        return torch.baddbmm(m, xc, M)

    def reconstruct_patches(
        self,
        vol: torch.Tensor,
        bases: torch.Tensor,
        patch_shape: tuple[int, ...],
        vh: torch.Tensor,
        ratio: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        wsum: torch.Tensor,
    ) -> None:
        """Weighted reconstruction of patches read in place, scattered in place.

        ``weight[b] * ((X[b] - m[b]) @ M[b] + m[b])`` is atomically added into
        ``out`` and ``weight[b]`` into ``wsum``, over the footprint of patch
        ``b``. ``out`` and ``wsum`` must share ``vol``'s shape and strides.
        See :meth:`eigh_patches` for ``vol``, ``bases`` and ``patch_shape``.
        """
        M = self._filter_matrix(vh, ratio)
        _fused_reconstruction(vol, bases, M, patch_shape, weight, out, wsum)

    def center_reconstruct(
        self,
        x_flat: torch.Tensor,
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
        vh : (B, k, T) tensor
            Right singular vectors, ``Vh`` row-convention (matches
            ``torch.linalg.svd``'s ``vh``). ``U`` is never computed
        ratio : (B, k) tensor
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
        return self.reconstruct_row(x_flat[:, spatial_idx, :], vh, ratio, time_idx)

    @staticmethod
    def reconstruct_row(
        x_row: torch.Tensor,
        vh: torch.Tensor,
        ratio: torch.Tensor,
        time_idx: int | None = None,
    ) -> torch.Tensor:
        """Reconstruct a single row ``x_row`` (B, T) of each patch.

        ``(x_row - m) @ V diag(ratio) Vh + m``, contracted right to left
        from the row: O(T k) per patch instead of the O(T^3) filter matrix.
        See :meth:`center_reconstruct` for ``time_idx`` and the output.
        """
        m = x_row.mean(dim=-1, keepdim=True)  # its temporal mean, (B, 1)
        coef = torch.einsum("bt,bkt->bk", x_row - m, vh.conj()) * ratio
        if time_idx is None:
            return torch.einsum("bk,bkt->bt", coef, vh) + m
        return torch.einsum("bk,bk->b", coef, vh[:, :, time_idx]) + m[:, 0]
