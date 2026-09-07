"""Tests for FastPatchSVD (``gpu/_svd.py``) against a frozen gesvda reference.

Unlike the rest of the GPU test suite, these tests instantiate
``OptimalSVDDenoiser``/``MPPCADenoiser`` directly rather than going through
``main_gpu`` -- they target the SVD-replacement boundary itself, mirroring how
``svd_optimization_mwe.py`` was validated before this landed: compare against
the real, unmodified ``gesvda``-based forward pass, not just internal
self-consistency (which previously hid a real bug -- the row/column
eigenvector convention issue documented in ``_svd.py``'s module docstring).

``gesvda`` is deliberately unreachable from production after this change, so
the reference forward functions below are frozen, verbatim copies of the
pre-change ``forward()`` bodies -- the only way left to validate the new path
against it.
"""

import pytest
import torch

from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.gpu._svd import FastPatchSVD
from patch_denoise.gpu.denoiser import MPPCADenoiser, OptimalSVDDenoiser

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")

PATCH_SHAPE = (5, 5, 5, 8)
BATCH_SIZE = 32
SEEDS = range(5)
DC_OFFSETS = (0.0, 50.0, 1000.0)
# Margin over the MWE's own observed worst-case relative reconstruction error
# (~1.9e-4, measured against real brain DWI data).
RECON_TOL = 1e-3


def _reference_optimal_forward(denoiser, x, var_apriori=None):
    """Frozen copy of ``OptimalSVDDenoiser.forward`` before the FastPatchSVD swap."""
    x_flat = x.reshape(x.shape[0], -1, x.shape[-1])
    n_dim = x_flat.shape[-2]
    m = torch.mean(x_flat, dim=-2, keepdim=True)
    xc = x_flat - m
    u, s, v = torch.linalg.svd(xc, full_matrices=False, driver="gesvda")

    if var_apriori is not None:
        sigma = torch.sqrt(var_apriori)
    else:
        n_svals = s.shape[-1]
        lo, hi = (n_svals - 1) // 2, n_svals // 2
        median_s = 0.5 * (s[..., lo] + s[..., hi])
        sigma = median_s / (denoiser.mp_median * (n_dim**0.5))
    scale_factor = sigma * (n_dim**0.5)

    scale_factor_exp = scale_factor.unsqueeze(-1)
    s_shrink = denoiser._shrink(s / scale_factor_exp)
    s_shrink = s_shrink * scale_factor_exp
    s_shrink = torch.nan_to_num(s_shrink, nan=0.0)

    maxidx = torch.sum(s_shrink > 0, dim=-1)

    if denoiser.recombination == "center":
        u_center = u[:, denoiser.center_spatial_idx, :]
        if denoiser.full_time:
            x_center = (
                torch.matmul((u_center * s_shrink).unsqueeze(1), v).squeeze(1)
                + m[:, 0, :]
            )
            return x_center, 1, sigma**2, maxidx.to(torch.int32)
        v_center = v[:, :, denoiser.center_time_idx]
        x_center = (
            torch.sum(u_center * s_shrink * v_center, dim=-1)
            + m[:, 0, denoiser.center_time_idx]
        )
        return x_center, 1, sigma**2, maxidx.to(torch.int32)

    if denoiser.recombination == "weighted":
        weight = 1.0 / (2.0 + maxidx)
    else:
        weight = torch.ones_like(maxidx, dtype=torch.float32)

    x_denoised = torch.matmul(u * s_shrink.unsqueeze(1), v) + m

    return (
        x_denoised.reshape(x.shape),
        weight,
        sigma**2,
        maxidx.to(torch.int32),
    )


def _reference_mppca_forward(denoiser, x):
    """Frozen copy of ``MPPCADenoiser.forward`` before the FastPatchSVD swap."""
    x_flat = x.reshape(x.shape[0], -1, x.shape[-1])

    xm = torch.mean(x_flat, dim=-2, keepdim=True)
    xc = x_flat - xm

    u, s, v = torch.linalg.svd(xc, full_matrices=False, driver="gesvda")

    N, M = x_flat.shape[-2], x_flat.shape[-1]
    eigs = s**2 / (N - 1)
    cum_eigs = torch.cumsum(eigs, dim=-1)
    rcum_eigs = eigs - cum_eigs + cum_eigs[:, -1:]

    p_range = torch.arange(M, device=x.device)
    mask = ((eigs - eigs[:, -1:]) * (M - p_range) * (N - p_range)) > (
        4 * rcum_eigs * (M * N) ** 0.5 * denoiser.threshold_scale**2
    )
    p = torch.sum(mask, dim=-1)
    eigs = eigs * (p_range < p.unsqueeze(-1))
    s_shrink = torch.sqrt(eigs * (N - 1))

    batch_idx = torch.arange(x_flat.shape[0], device=x.device)
    var_estimate = rcum_eigs[batch_idx, p] / (M - p)

    # NB: preserves a pre-existing, unrelated quirk -- unlike every other
    # return path (including OptimalSVDDenoiser's), this "center" branch does
    # not cast p to int32. Left as-is deliberately, matching the real code.
    if denoiser.recombination == "center":
        u_center = u[:, denoiser.center_spatial_idx, :]
        if denoiser.full_time:
            x_center = (
                torch.matmul((u_center * s_shrink).unsqueeze(1), v).squeeze(1)
                + xm[:, 0, :]
            )
            return x_center, 1, var_estimate, p
        v_center = v[:, :, denoiser.center_time_idx]
        x_center = (
            torch.sum(u_center * s_shrink * v_center, dim=-1)
            + xm[:, 0, denoiser.center_time_idx]
        )
        return x_center, 1, var_estimate, p

    x_denoised = torch.matmul(u * s_shrink.unsqueeze(1), v) + xm

    if denoiser.recombination == "weighted":
        weight = 1.0 / (2.0 + p)
    else:
        weight = torch.ones_like(p, dtype=torch.float32)

    return x_denoised.reshape(x.shape), weight, var_estimate, p.to(torch.int32)


def _make_x(dtype, dc_offset, seed):
    torch.manual_seed(seed)
    shape = (BATCH_SIZE, *PATCH_SHAPE)
    if dtype == torch.complex64:
        x = torch.complex(
            torch.randn(*shape, device="cuda"), torch.randn(*shape, device="cuda")
        )
    else:
        x = torch.randn(*shape, device="cuda", dtype=dtype)
    return x + dc_offset


@pytest.mark.parametrize("method", ["optimal-fro", "mp-pca"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("dc_offset", DC_OFFSETS)
def test_fast_patch_svd_matches_gesvda(method, dtype, dc_offset):
    """FastPatchSVD-based forward must match the frozen gesvda reference.

    Exact rank match and float32-precision-level reconstruction error, across
    a DC-offset sweep (the catastrophic-cancellation trap ``_svd.py``'s module
    docstring warns about) and both recombination modes -- mirroring the
    validation ``svd_optimization_mwe.py`` used before this was integrated.
    """
    for recombination, full_time in (
        ("weighted", False),
        ("center", False),
        ("center", True),
    ):
        kwargs = dict(
            patch_shape=PATCH_SHAPE, recombination=recombination, full_time=full_time
        )
        if method == "optimal-fro":
            denoiser = OptimalSVDDenoiser(loss="fro", **kwargs).cuda()
            ref_fwd = _reference_optimal_forward
        else:
            denoiser = MPPCADenoiser(**kwargs).cuda()
            ref_fwd = _reference_mppca_forward

        for seed in SEEDS:
            x = _make_x(dtype, dc_offset, seed)
            case = (
                f"method={method} dtype={dtype} dc_offset={dc_offset} "
                f"recombination={recombination} full_time={full_time} seed={seed}"
            )
            with torch.inference_mode():
                out_val, out_weight, out_var, out_rank = denoiser(x)
                ref_val, ref_weight, ref_var, ref_rank = ref_fwd(denoiser, x)

            assert torch.equal(out_rank, ref_rank), f"rank mismatch: {case}"

            if isinstance(out_weight, torch.Tensor):
                torch.testing.assert_close(out_weight, ref_weight, rtol=1e-4, atol=1e-6)
            else:
                assert out_weight == ref_weight

            torch.testing.assert_close(out_var, ref_var, rtol=1e-3, atol=1e-6)

            scale = max(x.abs().max().item(), 1.0)
            rel_diff = (out_val - ref_val).abs().max().item() / scale
            assert rel_diff < RECON_TOL, (
                f"reconstruction rel err {rel_diff:.2e}: {case}"
            )


def test_fast_patch_svd_uneven_batch_sizes():
    """One FastPatchSVD instance must handle two different batch sizes in a row.

    Unit-level complement to ``test_gpu.py::test_main_gpu_uneven_batch``,
    directly exercising the lazy (batch_size, dtype) -> solver cache.
    """
    T = PATCH_SHAPE[-1]
    svd = FastPatchSVD(T)
    for B in (BATCH_SIZE, BATCH_SIZE // 2):
        x_flat = torch.randn(B, 100, T, device="cuda")
        s, vh, m, xc = svd.eigh(x_flat)
        assert s.shape == (B, T)
        assert vh.shape == (B, T, T)
        assert xc is None  # float32 path never materializes the centered tensor

        ratio = torch.ones(B, T, device="cuda")
        out = svd.reconstruct(x_flat, m, xc, vh, ratio)
        assert out.shape == x_flat.shape
        assert torch.all(torch.isfinite(out))
