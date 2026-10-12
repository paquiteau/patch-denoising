"""Tests for FastPatchSVD (``gpu/_svd.py``): SVD of the patch matrix centered along time.

Compares ``FastPatchSVD.eigh``/``reconstruct``/``center_reconstruct`` directly
against a plain ``torch.linalg.svd(xc, driver="gesvda")`` reference, independent
of any denoiser.
"""

import itertools

import pytest

torch = pytest.importorskip("torch")

from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.gpu._svd import FastPatchSVD

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")

BATCH_SIZE = 32
DTYPES = (torch.float32, torch.complex64)
# DC offset: the catastrophic-cancellation trap ``_svd.py``'s module
# docstring warns about.
DC_OFFSETS = (0.0, 1000.0)
N = 100  # spatial extent of a patch
T_VALUES = (4, 8)  # channel/time dimension
# Margin over float32-precision-level error.
TOL = 1e-3


def _x_id(param):
    dtype, dc_offset, T = param
    dtype_name = "complex64" if dtype == torch.complex64 else "float32"
    return f"{dtype_name}-dc{dc_offset:g}-T{T}"


@pytest.fixture(
    params=list(itertools.product(DTYPES, DC_OFFSETS, T_VALUES)),
    ids=_x_id,
)
def x(request):
    dtype, dc_offset, T = request.param
    torch.manual_seed(0)
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


def _temporal_mean(x):
    return torch.mean(x, dim=-1, keepdim=True)


def _k(x):
    """Return the number of non-trivial components after centering along time."""
    return min(x.shape[-2], x.shape[-1] - 1)


def _sorted_ratio(x, seed):
    torch.manual_seed(seed)
    return (
        torch.rand(x.shape[0], _k(x), device="cuda")
        .sort(dim=-1, descending=True)
        .values
    )


def test_eigh_mean_and_singular_values_match_reference(x):
    """``eigh``'s mean and singular values must match a plain ``torch.linalg.svd``.

    Patches are centered along time, so the constant time course is an exact
    null direction: it is dropped by ``eigh`` and ~0 in the reference.
    """
    m_ref = _temporal_mean(x)
    xc_ref = x - m_ref

    s_ref = torch.linalg.svd(xc_ref, full_matrices=False, driver="gesvda")[1]
    k = _k(x)

    fastsvd = FastPatchSVD(x.shape[-1])
    with torch.inference_mode():
        s, _, m, _ = fastsvd.eigh(x)

    assert _rel_err(m, m_ref) < TOL, "mean mismatch"
    assert s.shape == (x.shape[0], k)
    assert _rel_err(s, s_ref[:, :k]) < TOL, "singular value mismatch"
    assert _rel_err(s_ref[:, k:], torch.zeros_like(s_ref[:, k:]), s_ref.max()) < TOL


def test_eigh_vh_is_unitary(x):
    """Each patch's ``Vh`` must have ``k`` orthonormal rows, all orthogonal to 1."""
    k, T = _k(x), x.shape[-1]
    eye = torch.eye(k, device="cuda", dtype=x.dtype).expand(x.shape[0], k, k)
    fastsvd = FastPatchSVD(T)
    with torch.inference_mode():
        _, vh, _, _ = fastsvd.eigh(x)
    gram = torch.matmul(vh, _hermitian_t(vh))
    assert _rel_err(gram, eye, scale=1.0) < TOL
    ones = torch.ones(T, 1, device="cuda", dtype=x.dtype)
    assert _rel_err(vh @ ones, torch.zeros_like(vh @ ones), scale=1.0) < TOL


def test_eigh_reconstructs_centered_gram_matrix(x):
    """``Vh^H diag(s^2) Vh`` must equal the centered Gram matrix ``Xc^H Xc``."""
    xc_ref = x - _temporal_mean(x)
    gram_ref = torch.matmul(_hermitian_t(xc_ref), xc_ref)

    fastsvd = FastPatchSVD(x.shape[-1])
    with torch.inference_mode():
        s, vh, _, _ = fastsvd.eigh(x)
    v = _hermitian_t(vh)
    s2 = (s**2).to(v.dtype)
    gram = torch.matmul(v * s2.unsqueeze(-2), vh)

    assert _rel_err(gram, gram_ref) < TOL


def test_reconstruct_matches_reference_at_arbitrary_ratio(x):
    """Partial-rank reconstruction must match a plain full ``torch.linalg.svd`` filter.

    The filter matrix ``V @ diag(ratio) @ Vh`` is invariant to each singular
    vector's arbitrary sign/phase, so this holds even though FastPatchSVD
    never computes the same ``U``/``V`` as the reference. ``ratio`` is sorted
    descending (matching how a real shrink function assigns nearby ratios to
    nearby singular values): an i.i.d. random ratio can assign very different
    shrinkage to a pair of near-degenerate singular values, which is
    genuinely ill-conditioned -- the eigenvectors spanning a near-degenerate
    subspace aren't well-defined, so two independent, individually-correct
    solvers (this one and ``gesvda``) can legitimately disagree there.
    """
    ratio = _sorted_ratio(x, 1234)
    k = _k(x)

    m_ref = _temporal_mean(x)
    xc_ref = x - m_ref
    u_ref, s_ref, vh_ref = torch.linalg.svd(
        xc_ref, full_matrices=False, driver="gesvda"
    )
    ref = (
        torch.matmul(
            u_ref[..., :k] * (s_ref[:, :k] * ratio).unsqueeze(-2), vh_ref[:, :k]
        )
        + m_ref
    )

    fastsvd = FastPatchSVD(x.shape[-1])
    with torch.inference_mode():
        _, vh, m, xc = fastsvd.eigh(x)
        out = fastsvd.reconstruct(x, m, xc, vh, ratio)

    assert _rel_err(out, ref) < TOL


@pytest.mark.parametrize("full_time", [False, True])
def test_center_reconstruct_matches_reconstruct_row(x, full_time):
    """``center_reconstruct`` must equal the corresponding row of ``reconstruct``."""
    spatial_idx = x.shape[-2] // 2
    time_idx = None if full_time else x.shape[-1] // 2

    ratio = _sorted_ratio(x, 1)
    fastsvd = FastPatchSVD(x.shape[-1])
    with torch.inference_mode():
        s, vh, m, xc = fastsvd.eigh(x)
        full = fastsvd.reconstruct(x, m, xc, vh, ratio)
        center = fastsvd.center_reconstruct(x, vh, ratio, spatial_idx, time_idx)

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
        assert s.shape == (B, T - 1)
        assert vh.shape == (B, T - 1, T)

        ratio = torch.ones(B, T - 1, device="cuda")
        out = svd.reconstruct(x_flat, m, xc, vh, ratio)
        assert out.shape == x_flat.shape
        assert torch.all(torch.isfinite(out))


@pytest.mark.parametrize("dtype", DTYPES)
def test_eigh_resolves_noise_floor_under_strong_mean_image(dtype):
    """FP32 must resolve the noise eigenvalues of patches with a dominant mean image.

    Real patches carry a spatially varying baseline (the image contrast) that
    is constant over time. Centering over voxels would leave it in the Gram
    matrix as a component ~1e5 times the noise eigenvalues, and an FP32
    eigensolver (absolute error ~eps * lambda_1) then gets the noise
    eigenvalues wrong by percents. Centering along time removes it.
    """
    torch.manual_seed(0)
    B, N, T = 64, 1331, 48
    baseline = 1000.0 * (1.2 + torch.rand(B, N, 1, device="cuda"))
    signal = (
        20.0 * torch.randn(B, N, 3, device="cuda") @ torch.randn(B, 3, T, device="cuda")
    )
    noise = 4.0 * torch.randn(B, N, T, device="cuda")
    x = (baseline + signal + noise).to(dtype)

    x64 = x.to(torch.complex128 if dtype == torch.complex64 else torch.float64)
    s_ref = torch.linalg.svdvals(x64 - _temporal_mean(x64))[:, : T - 1]

    fastsvd = FastPatchSVD(T)
    with torch.inference_mode():
        s, *_ = fastsvd.eigh(x)
    rel = ((s.double() - s_ref) / s_ref).abs()
    assert rel.max().item() < 1e-3


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", [8, 40, 100, 130])  # one or several T tiles
def test_patches_in_place_match_batch(dtype, T):
    """``eigh_patches``/``reconstruct_patches`` must match ``eigh``/``reconstruct``.

    Patches are disjoint (5, 5, 5, T) blocks of a volume, shifted by the
    temporal mean of each voxel as ``main_gpu`` does, with a strong mean
    image: the in-place path centers patches algebraically.
    """
    torch.manual_seed(0)
    B, p = 16, 5
    shape = (B * p, p, p, T)
    vol = torch.randn(shape, device="cuda", dtype=dtype)
    vol += 1000.0 * (1.2 + torch.rand(*shape[:-1], 1, device="cuda"))
    x_flat = vol.reshape(B, p**3, T)
    bases = torch.arange(B, device="cuda") * (p * vol.stride(0))
    shifted = vol - vol.mean(-1, keepdim=True)
    ratio = _sorted_ratio(x_flat, 0)
    weight = torch.rand(B, device="cuda") + 0.5

    fastsvd = FastPatchSVD(T)
    with torch.inference_mode():
        s_ref, vh_ref, m, xc = fastsvd.eigh(x_flat)
        ref = fastsvd.reconstruct(x_flat, m, xc, vh_ref, ratio) - m
        s, vh = fastsvd.eigh_patches(shifted, bases, (p, p, p, T))
        out = torch.zeros_like(vol)
        wsum = torch.zeros_like(vol, dtype=torch.float32)
        fastsvd.reconstruct_patches(
            shifted, bases, (p, p, p, T), vh, ratio, weight, out, wsum
        )

    assert _rel_err(s, s_ref) < TOL
    w = weight[:, None, None]
    assert _rel_err(out.reshape(B, p**3, T) / w, ref) < TOL
    assert _rel_err(wsum.reshape(B, p**3, T), w.expand(B, p**3, T)) == 0
