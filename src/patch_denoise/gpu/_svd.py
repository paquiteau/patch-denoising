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
- Patches are centered along time (``m`` is the temporal mean of each voxel):
  ``Xc = X P`` with ``P = I - 11^T / T``. This makes the constant time course
  an exact null direction of ``Xc``: only the ``k = min(N, T - 1)``
  non-trivial components are returned. It also removes the mean image -- by
  far the dominant component of a real patch -- from the Gram matrix, which
  keeps it well conditioned enough for an FP32 eigensolver to resolve the
  noise eigenvalues.

Patches read in place from a volume (:meth:`FastPatchSVD.eigh_patches`) are
centered algebraically, so that the Triton kernels can tile the time axis:

- The Gram matrix is ``Xc^H Xc = P (X^H X) P``: a plain tiled ``X^H X``,
  projected on the small ``(T, T)`` matrix.
- The reconstruction is ``Xc @ M + m = X @ (M + 11^T / T)``, since ``V`` is
  orthogonal to ``1``: one tiled GEMM, without the mean.

Both are exact, but ``X^H X`` carries the mean image again. The volume must
thus first be shifted by any per-voxel constant close to its temporal mean
(e.g. the mean over the whole time axis, see ``main_gpu``): ``P`` cancels it
exactly, and the shifted Gram matrix is as well conditioned as the centered one.
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
        # On GPU to avoid host-side check of failure.
        self.n_failed = torch.zeros((), device=device, dtype=torch.int64)

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
        self.n_failed += torch.count_nonzero(self.info)
        return w


@triton.jit
def _patch_offsets(base, n, sx, sy, sz, PY: tl.constexpr, PZ: tl.constexpr):
    """Element offset of voxel ``n`` (C order) of a ``(PX, PY, PZ)`` patch."""
    return base + (n // (PY * PZ)) * sx + (n // PZ % PY) * sy + (n % PZ) * sz


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": bt, "BLOCK_N": 32}, num_warps=4, num_stages=s)
        for bt in (32, 64)
        for s in (2, 3)
    ],
    key=["T", "N", "IS_COMPLEX"],
)
@triton.jit
def _gram_kernel(
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
    IS_COMPLEX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute the Gram matrix ``out[b] = X[b]^H X[b]`` of a batch of patches.

    ``X[b]`` is the ``(N, T)`` patch whose first element is at
    ``x_ptr + base[b]``: patches are read in place, never gathered. Each
    program computes one ``(BLOCK_T, BLOCK_T)`` tile of the lower triangle,
    and mirrors it to the upper one. Complex data is read as interleaved
    float32, with strides counted in complex elements.
    """
    pid_b = tl.program_id(0).to(tl.int64)
    # Linear index of a lower-triangle tile -> (ti, tj), ti >= tj.
    p = tl.program_id(1)
    ti = ((tl.sqrt(8.0 * p + 1.0) - 1.0) * 0.5).to(tl.int32)
    ti = tl.where((ti + 1) * (ti + 2) // 2 <= p, ti + 1, ti)
    ti = tl.where(ti * (ti + 1) // 2 > p, ti - 1, ti)
    tj = p - ti * (ti + 1) // 2

    base = tl.load(base_ptr + pid_b)
    ri = ti * BLOCK_T + tl.arange(0, BLOCK_T)
    rj = tj * BLOCK_T + tl.arange(0, BLOCK_T)
    acc_r = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        rows = _patch_offsets(base, n_idx, sx, sy, sz, PY, PZ)
        mask_i = (n_idx < N)[:, None] & (ri < T)[None, :]
        mask_j = (n_idx < N)[:, None] & (rj < T)[None, :]
        off_i = rows[:, None] + ri[None, :] * st
        off_j = rows[:, None] + rj[None, :] * st
        if IS_COMPLEX:
            ar = tl.trans(tl.load(x_ptr + 2 * off_i, mask=mask_i, other=0.0))
            ai = tl.trans(tl.load(x_ptr + 2 * off_i + 1, mask=mask_i, other=0.0))
            br = tl.load(x_ptr + 2 * off_j, mask=mask_j, other=0.0)
            bi = tl.load(x_ptr + 2 * off_j + 1, mask=mask_j, other=0.0)
            acc_r = tl.dot(ar, br, acc_r, input_precision="ieee")
            acc_r = tl.dot(ai, bi, acc_r, input_precision="ieee")
            acc_i = tl.dot(ar, bi, acc_i, input_precision="ieee")
            acc_i = tl.dot(-ai, br, acc_i, input_precision="ieee")
        else:
            a = tl.load(x_ptr + off_i, mask=mask_i, other=0.0)
            b = tl.load(x_ptr + off_j, mask=mask_j, other=0.0)
            acc_r = tl.dot(tl.trans(a), b, acc_r, input_precision="ieee")

    out_mask = (ri < T)[:, None] & (rj < T)[None, :]
    out_off = pid_b * T * T + ri[:, None] * T + rj[None, :]
    mirror_off = pid_b * T * T + ri[:, None] + rj[None, :] * T
    if IS_COMPLEX:
        tl.store(out_ptr + 2 * out_off, acc_r, mask=out_mask)
        tl.store(out_ptr + 2 * out_off + 1, acc_i, mask=out_mask)
        if ti != tj:
            tl.store(out_ptr + 2 * mirror_off, acc_r, mask=out_mask)
            tl.store(out_ptr + 2 * mirror_off + 1, -acc_i, mask=out_mask)
    else:
        tl.store(out_ptr + out_off, acc_r, mask=out_mask)
        if ti != tj:
            tl.store(out_ptr + mirror_off, acc_r, mask=out_mask)


# ``restore_value``: autotuning runs the kernel several times, which must not
# accumulate into the outputs more than once.
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_T": bt, "BLOCK_K": 32}, num_warps=4, num_stages=s
        )
        for bt in (32, 64)
        for s in (2, 3)
    ],
    key=["T", "N", "IS_COMPLEX"],
    restore_value=["out_ptr", "wsum_ptr"],
)
@triton.jit
def _scatter_reconstruction_kernel(
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
    IS_COMPLEX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scatter the weighted reconstruction ``w[b] * X[b] @ Mm[b]`` in place.

    Patches are addressed as in the Gram kernel. Each program computes one
    ``(BLOCK_N, BLOCK_T)`` tile, atomically added into ``out`` (which shares
    ``x``'s layout), and ``w[b]`` into ``wsum``: the reconstructed batch never
    exists in memory.
    """
    pid_b = tl.program_id(0).to(tl.int64)
    w = tl.load(w_ptr + pid_b)
    n_idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    t_idx = tl.program_id(2) * BLOCK_T + tl.arange(0, BLOCK_T)
    rows = _patch_offsets(tl.load(base_ptr + pid_b), n_idx, sx, sy, sz, PY, PZ)
    n_mask = n_idx < N
    t_mask = t_idx < T

    acc_r = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for k0 in range(0, T, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < T
        x_off = rows[:, None] + k_idx[None, :] * st
        x_mask = n_mask[:, None] & k_mask[None, :]
        mm_off = pid_b * T * T + k_idx[:, None] * T + t_idx[None, :]
        mm_mask = k_mask[:, None] & t_mask[None, :]
        if IS_COMPLEX:
            xr = tl.load(x_ptr + 2 * x_off, mask=x_mask, other=0.0)
            xi = tl.load(x_ptr + 2 * x_off + 1, mask=x_mask, other=0.0)
            mr = tl.load(mm_ptr + 2 * mm_off, mask=mm_mask, other=0.0)
            mi = tl.load(mm_ptr + 2 * mm_off + 1, mask=mm_mask, other=0.0)
            acc_r = tl.dot(xr, mr, acc_r, input_precision="ieee")
            acc_r = tl.dot(-xi, mi, acc_r, input_precision="ieee")
            acc_i = tl.dot(xr, mi, acc_i, input_precision="ieee")
            acc_i = tl.dot(xi, mr, acc_i, input_precision="ieee")
        else:
            x = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
            mm = tl.load(mm_ptr + mm_off, mask=mm_mask, other=0.0)
            acc_r = tl.dot(x, mm, acc_r, input_precision="ieee")

    offsets = rows[:, None] + t_idx[None, :] * st
    mask2d = n_mask[:, None] & t_mask[None, :]
    if IS_COMPLEX:
        tl.atomic_add(out_ptr + 2 * offsets, w * acc_r, mask=mask2d, sem="relaxed")
        tl.atomic_add(out_ptr + 2 * offsets + 1, w * acc_i, mask=mask2d, sem="relaxed")
    else:
        tl.atomic_add(out_ptr + offsets, w * acc_r, mask=mask2d, sem="relaxed")
    tl.atomic_add(
        wsum_ptr + offsets, w + tl.zeros_like(acc_r), mask=mask2d, sem="relaxed"
    )


def _as_real(x: torch.Tensor) -> torch.Tensor:
    """View complex tensors as interleaved float32, for the kernels above."""
    return torch.view_as_real(x) if x.is_complex() else x


def _patch_layout(vol, patch_shape):
    """Kernel addressing arguments: ``(T, sx, sy, sz, st)`` and ``(N, PY, PZ)``."""
    px, py, pz, T = patch_shape
    return (T, *vol.stride()), dict(N=px * py * pz, PY=py, PZ=pz)


def _project_time_(a: torch.Tensor) -> torch.Tensor:
    """Compute ``P a P`` in place, ``P = I - 11^T / T``, on a (B, T, T) batch."""
    row = a.mean(-1, keepdim=True)
    col = a.mean(-2, keepdim=True)
    return a.sub_(row).sub_(col).add_(row.mean(-2, keepdim=True))


def _gram_patches(vol, bases, patch_shape):
    """Compute the batched centered Gram matrix of patches read in place.

    Parameters
    ----------
    vol : (X, Y, Z, T') tensor, float32 or complex64
    bases : (B,) int64 tensor
        Element offset in ``vol`` of each patch's first voxel.
    patch_shape : (px, py, pz, T) tuple

    Returns
    -------
    gc : (B, T, T) tensor, same dtype as ``vol``
        ``gc[b] = Xc[b]^H Xc[b]``, ``Xc[b] = X[b] P``, the patch centered
        along time.
    """
    (T, *strides), meta = _patch_layout(vol, patch_shape)
    B = bases.shape[0]
    gc = torch.empty(B, T, T, device=vol.device, dtype=vol.dtype)

    def grid(cfg):
        n_tiles = triton.cdiv(T, cfg["BLOCK_T"])
        return (B, n_tiles * (n_tiles + 1) // 2)

    _gram_kernel[grid](
        _as_real(vol),
        bases,
        _as_real(gc),
        T,
        *strides,
        IS_COMPLEX=vol.is_complex(),
        **meta,
    )
    return _project_time_(gc)


def _scatter_reconstruction(vol, bases, patch_shape, Mm, weight, out, wsum):
    """Atomically add the weighted reconstructions ``weight[b] * X[b] @ Mm[b]``.

    Into ``out``, and ``weight`` into ``wsum``, which must share the shape and
    strides of ``vol``.
    """
    if out.stride() != vol.stride() or wsum.stride() != vol.stride():
        raise ValueError("`out` and `wsum` must share the layout of `vol`.")
    (T, *strides), meta = _patch_layout(vol, patch_shape)
    B = bases.shape[0]

    def grid(cfg):
        return (
            B,
            triton.cdiv(meta["N"], cfg["BLOCK_N"]),
            triton.cdiv(T, cfg["BLOCK_T"]),
        )

    _scatter_reconstruction_kernel[grid](
        _as_real(vol),
        bases,
        _as_real(Mm.contiguous()),
        weight.to(torch.float32),
        _as_real(out),
        wsum,
        T,
        *strides,
        IS_COMPLEX=vol.is_complex(),
        **meta,
    )


class FastPatchSVD(torch.nn.Module):
    """
    Fast batched SVD/eigensolve for patch denoising, replacing ``gesvda``.

    This class implements a custom SVD driver for a batch of tall-skinny
    matrices (patches) that avoids the FP64 promotion of ``gesvda`` in cuSOLVER.

    It computes the eigendecomposition of the Gram matrix of the patches,
    centered along time, and reconstructs the denoised patches without
    explicitly forming the left singular vectors.

    Patches are either given as a ``(B, N, T)`` batch (:meth:`eigh`,
    :meth:`reconstruct`), or read in place from a volume by the Triton kernels
    (:meth:`eigh_patches`, :meth:`reconstruct_patches`), which never
    materialize the patches nor their reconstruction.

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

    def n_failed(self) -> int:
        """Count the patches the eigensolver failed to converge on, so far.

        Syncs with the GPU: meant to be checked once, after the last batch.
        """
        return int(sum(s.n_failed for s in self._solvers.values()))

    def eigh(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        m : (B, N, 1) tensor
            Temporal mean of each voxel.
        xc : (B, N, T) tensor
            Centered data, reused by :meth:`reconstruct`.
        """
        dtype = x_flat.dtype
        if dtype not in _CUDA_DTYPE:
            raise TypeError(
                f"FastPatchSVD only supports float32/complex64, got {dtype}"
            )
        m = torch.mean(x_flat, dim=-1, keepdim=True)
        xc = x_flat - m
        gc = torch.matmul(xc.mH, xc)
        s, vh = self._solve(gc, x_flat.shape[-2])
        return s, vh, m, xc

    def eigh_patches(
        self, vol: torch.Tensor, bases: torch.Tensor, patch_shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eigendecompose, as :meth:`eigh`, patches read in place from ``vol``.

        For accuracy, ``vol`` should be shifted by (about) the temporal mean of
        each voxel, see the module docstring.

        Parameters
        ----------
        vol : (X, Y, Z, T') tensor, float32 or complex64
        bases : (B,) int64 tensor
            Element offset in ``vol`` of the first voxel of each patch.
        patch_shape : (px, py, pz, T) tuple

        Returns
        -------
        s, vh :
            As for :meth:`eigh`.
        """
        gc = _gram_patches(vol, bases, patch_shape)
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
        return torch.matmul(vh.mH * ratio.unsqueeze(-2), vh)

    def reconstruct(
        self,
        x_flat: torch.Tensor,
        m: torch.Tensor,
        xc: torch.Tensor,
        vh: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Full (B, N, T) reconstruction: ``Xc @ M + m``, see :meth:`eigh`."""
        return torch.baddbmm(m, xc, self._filter_matrix(vh, ratio))

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
        # (X - m) @ M + m == X @ (M + 11^T / T), see the module docstring.
        Mm = self._filter_matrix(vh, ratio).add_(1.0 / patch_shape[-1])
        _scatter_reconstruction(vol, bases, patch_shape, Mm, weight, out, wsum)

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
