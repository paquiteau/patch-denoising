"""Tests for FastPatchSVD (``gpu/_svd.py``): SVD of the centered patch matrix.

Compares ``FastPatchSVD.eigh``/``reconstruct``/``center_reconstruct`` directly
against a plain ``torch.linalg.svd(xc, driver="gesvda")`` reference, independent
of any denoiser.
"""

import itertools

import pytest
import torch

from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.gpu._svd import FastPatchSVD

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")

BATCH_SIZE = 32
SEEDS = range(5)
DTYPES = (torch.float32, torch.complex64)
# DC-offset sweep: the catastrophic-cancellation trap ``_svd.py``'s module
# docstring warns about.
DC_OFFSETS = (0.0, 50.0, 1000.0)
N_VALUES = (50, 100)  # spatial extent of a patch, incl. N < T (rank-deficient)
T_VALUES = (4, 8)  # channel/time dimension
# Margin over float32-precision-level error.
TOL = 1e-3


def _x_id(param):
    dtype, dc_offset, seed, N, T = param
    dtype_name = "complex64" if dtype == torch.complex64 else "float32"
    return f"{dtype_name}-dc{dc_offset:g}-seed{seed}-N{N}-T{T}"


@pytest.fixture(
    params=list(itertools.product(DTYPES, DC_OFFSETS, SEEDS, N_VALUES, T_VALUES)),
    ids=_x_id,
)
def x(request):
    dtype, dc_offset, seed, N, T = request.param
    torch.manual_seed(seed)
    if dtype == torch.complex64:
        x = torch.complex(
            torch.randn(BATCH_SIZE, N, T, device="cuda"),
            torch.randn(BATCH_SIZE, N, T, device="cuda"),
        )
    else:
        x = torch.randn(BATCH_SIZE, N, T, device="cuda", dtype=dtype)
    return x + dc_offset


def _hermitian_t(a):
    return a.conj().transpose(-2, -1)


def _rel_err(actual, expected, scale=None):
    if scale is None:
        scale = max(expected.abs().max().item(), 1.0)
    return (actual - expected).abs().max().item() / scale


def test_eigh_mean_and_singular_values_match_reference(x):
    """``eigh``'s mean and singular values must match a plain ``torch.linalg.svd``."""
    m_ref = torch.mean(x, dim=-2, keepdim=True)
    xc_ref = x - m_ref

    s_ref = torch.linalg.svd(xc_ref, full_matrices=False, driver="gesvda")[1]

    fastsvd = FastPatchSVD(x.shape[-1])
    with torch.inference_mode():
        s, _, m, _ = fastsvd.eigh(x)

    assert _rel_err(m, m_ref) < TOL, "mean mismatch"
    assert _rel_err(s, s_ref) < TOL, "singular value mismatch"


def test_eigh_vh_is_unitary(svd, x):
    """Each patch's ``Vh`` must be a unitary ``(T, T)`` matrix."""
    T = x.shape[-1]
    eye = torch.eye(T, device="cuda", dtype=x.dtype).expand(x.shape[0], T, T)
    with torch.inference_mode():
        _, vh, _, _ = svd.eigh(x)
    gram = torch.matmul(vh, _hermitian_t(vh))
    assert _rel_err(gram, eye, scale=1.0) < TOL


def test_eigh_reconstructs_centered_gram_matrix(svd, x):
    """``Vh^H diag(s^2) Vh`` must equal the centered Gram matrix ``Xc^H Xc``."""
    xc_ref = x - torch.mean(x, dim=-2, keepdim=True)
    gram_ref = torch.matmul(_hermitian_t(xc_ref), xc_ref)

    with torch.inference_mode():
        s, vh, _, _ = svd.eigh(x)
    v = _hermitian_t(vh)
    s2 = (s**2).to(v.dtype)
    gram = torch.matmul(v * s2.unsqueeze(-2), vh)

    assert _rel_err(gram, gram_ref) < TOL


def test_reconstruct_full_rank_is_identity(svd, x):
    """``ratio=1`` for every component must round-trip back to the original patches."""
    with torch.inference_mode():
        s, vh, m, xc = svd.eigh(x)
        ratio = torch.ones_like(s)
        out = svd.reconstruct(x, m, xc, vh, ratio)
    assert _rel_err(out, x) < TOL


def test_reconstruct_zero_rank_returns_mean(svd, x):
    """``ratio=0`` for every component must collapse each patch to its own mean."""
    with torch.inference_mode():
        s, vh, m, xc = svd.eigh(x)
        ratio = torch.zeros_like(s)
        out = svd.reconstruct(x, m, xc, vh, ratio)
    assert _rel_err(out, m.expand_as(out)) < TOL


def test_reconstruct_matches_reference_at_arbitrary_ratio(svd, x):
    """Partial-rank reconstruction must match a plain full ``torch.linalg.svd`` filter.

    The filter matrix ``V @ diag(ratio) @ Vh`` is invariant to each singular
    vector's arbitrary sign/phase, so this holds even though FastPatchSVD
    never computes the same ``U``/``V`` as the reference.
    """
    torch.manual_seed(1234)
    ratio = torch.rand(x.shape[0], x.shape[-1], device="cuda")

    m_ref = torch.mean(x, dim=-2, keepdim=True)
    xc_ref = x - m_ref
    u_ref, s_ref, vh_ref = torch.linalg.svd(
        xc_ref, full_matrices=False, driver="gesvda"
    )
    ref = torch.matmul(u_ref * (s_ref * ratio).unsqueeze(-2), vh_ref) + m_ref

    with torch.inference_mode():
        _, vh, m, xc = svd.eigh(x)
        out = svd.reconstruct(x, m, xc, vh, ratio)

    assert _rel_err(out, ref) < TOL


@pytest.mark.parametrize("full_time", [False, True])
def test_center_reconstruct_matches_reconstruct_row(svd, x, full_time):
    """``center_reconstruct`` must equal the corresponding row of ``reconstruct``."""
    spatial_idx = x.shape[-2] // 2
    time_idx = None if full_time else x.shape[-1] // 2

    torch.manual_seed(1)
    ratio = torch.rand(x.shape[0], x.shape[-1], device="cuda")

    with torch.inference_mode():
        s, vh, m, xc = svd.eigh(x)
        full = svd.reconstruct(x, m, xc, vh, ratio)
        center = svd.center_reconstruct(x, m, vh, ratio, spatial_idx, time_idx)

    expected = full[:, spatial_idx, :] if full_time else full[:, spatial_idx, time_idx]
    assert _rel_err(center, expected) < TOL


def test_fast_patch_svd_uneven_batch_sizes():
    """One FastPatchSVD instance must handle two different batch sizes in a row.

    Directly exercises the lazy (batch_size, dtype) -> solver cache.
    """
    N, T = 100, 8
    svd = FastPatchSVD(T)
    for B in (BATCH_SIZE, BATCH_SIZE // 2):
        x_flat = torch.randn(B, N, T, device="cuda")
        s, vh, m, xc = svd.eigh(x_flat)
        assert s.shape == (B, T)
        assert vh.shape == (B, T, T)
        assert xc is None  # float32 path never materializes the centered tensor

        ratio = torch.ones(B, T, device="cuda")
        out = svd.reconstruct(x_flat, m, xc, vh, ratio)
        assert out.shape == x_flat.shape
        assert torch.all(torch.isfinite(out))
