"""Tests for the GPU denoising pipeline."""

import numpy as np
import numpy.testing as npt
import pytest

from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.denoise import mp_pca, optimal_thresholding
from patch_denoise.simulation.activations import add_frames
from patch_denoise.simulation.noise import add_temporal_gaussian_noise
from patch_denoise.simulation.phantom import g_factor_map, mr_shepp_logan_t2_star
from patch_denoise.space_time.base import ExtraOutput

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")

PATCH_SHAPE = (6, 6, 6, 8)
PATCH_OVERLAP = (3, 3, 3, 0)
GPU_METHODS = ["mp-pca", "optimal-fro", "optimal-nuc", "optimal-ope"]


def _method_kwargs(method):
    """Extra denoising kwargs that only apply to a given method."""
    return {"threshold_scale": 2.3} if method == "mp-pca" else {}


@pytest.fixture(scope="module")
def noisy_phantom_4d(rng):
    """Create a small 4D (X, Y, Z, T) noisy phantom, real and imaginary parts."""
    phantom = add_frames(mr_shepp_logan_t2_star(24), 8)
    g_map = g_factor_map(phantom.shape[:-1])
    real = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    imag = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    return phantom, real, imag


@pytest.mark.parametrize("method", GPU_METHODS)
def test_main_gpu_complex64(noisy_phantom_4d, method):
    from patch_denoise.gpu.main import main_gpu

    phantom, real, imag = noisy_phantom_4d
    data = (real + 1j * imag).astype(np.complex64)
    mask = np.ones(data.shape[:-1], dtype=bool)

    denoised, weights, noise_std_map, _, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method=method,
        mask=mask,
        batch_size=8,
        **_method_kwargs(method),
    )

    assert denoised.shape == data.shape
    assert denoised.dtype == np.complex64
    assert np.all(np.isfinite(denoised))

    # a correctly-sized accumulation buffer should actually denoise the phantom
    noise_std_before = np.sqrt(np.nanmean(np.nanvar(np.abs(data - phantom), axis=-1)))
    noise_std_after = np.sqrt(
        np.nanmean(np.nanvar(np.abs(denoised - phantom), axis=-1))
    )
    assert noise_std_after < noise_std_before


@pytest.mark.parametrize("method", GPU_METHODS)
def test_main_gpu_matches_cpu(noisy_phantom_4d, method):
    """GPU denoising should match the CPU reference implementation.

    Uses a batch size that does not divide the number of temporal features
    (8), so a regression of the per-batch-row rank-selection bugs would
    show up as a crash rather than just a numerical mismatch.
    """
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)

    if method == "mp-pca":
        cpu_denoised, _, cpu_noise_std, _ = mp_pca(
            data,
            patch_shape=PATCH_SHAPE,
            patch_overlap=PATCH_OVERLAP,
            mask=mask,
            recombination="weighted",
            threshold_scale=2.3,
        )
    else:
        cpu_denoised, _, cpu_noise_std, _ = optimal_thresholding(
            data,
            patch_shape=PATCH_SHAPE,
            patch_overlap=PATCH_OVERLAP,
            mask=mask,
            recombination="weighted",
            loss=method.split("-")[-1],
        )

    gpu_denoised, _, gpu_noise_std, _, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method=method,
        mask=mask,
        batch_size=5,
        extra_output=ExtraOutput.NOISE_STD,
        **_method_kwargs(method),
    )

    npt.assert_allclose(gpu_denoised, cpu_denoised, rtol=1e-5, atol=1e-2)

    # Both backends should report a standard-deviation map, at the full
    # (X, Y, Z, T) shape of the input (not a variance, and not reduced
    # over any axis).
    assert gpu_noise_std.shape == cpu_noise_std.shape == data.shape
    npt.assert_allclose(gpu_noise_std, cpu_noise_std, rtol=1e-5, atol=1e-2)


def test_main_gpu_uneven_batch(noisy_phantom_4d):
    """The last, smaller batch of a run must not crash or produce NaN.

    Regression guard for FastPatchSVD's fixed-workspace-per-batch-size
    solver: a batch_size that doesn't evenly divide the patch count forces
    the denoiser to see two different batch sizes in one run, which must
    lazily construct (and correctly use) a second eigensolver workspace
    sized for the remainder, alongside the main one.
    """
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)

    denoised, _, noise_std, _, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method="optimal-fro",
        mask=mask,
        batch_size=5,  # deliberately does not divide the patch count
        extra_output=ExtraOutput.NOISE_STD,
    )

    assert np.all(np.isfinite(denoised))
    assert np.all(np.isfinite(noise_std))


def test_main_gpu_optimal_fro_noise_matches_cpu(noisy_phantom_4d):
    """``optimal-fro-noise`` (an explicit noise map) must run and match CPU.

    Regression test: this used to crash with a TypeError (GPU's
    OptimalSVDDenoiser didn't accept noise_std, and the loss was
    mis-parsed as "noise" instead of "fro" from the method name).
    """
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)
    # Deliberately non-uniform so a spatially-wrong noise map (e.g. a stray
    # transpose or a uniform fallback) would show up as a mismatch.
    noise_std_map = (
        (1.0 + 0.1 * np.arange(np.prod(data.shape[:-1])))
        .reshape(data.shape[:-1])
        .astype(np.float32)
    )

    cpu_denoised, _, cpu_noise_std, _ = optimal_thresholding(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask=mask,
        recombination="weighted",
        loss="fro",
        noise_std=noise_std_map,
    )

    gpu_denoised, _, gpu_noise_std, _, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method="optimal-fro-noise",
        mask=mask,
        noise_std=noise_std_map,
        batch_size=5,
        extra_output=ExtraOutput.NOISE_STD,
    )

    assert np.all(np.isfinite(gpu_denoised))
    npt.assert_allclose(gpu_denoised, cpu_denoised, rtol=1e-5, atol=1e-2)
    npt.assert_allclose(gpu_noise_std, cpu_noise_std, rtol=1e-5, atol=1e-2)


def test_main_gpu_rank_map(noisy_phantom_4d):
    """The rank map should be real, not the placeholder None."""
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)

    _, _, _, rank_map, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method="optimal-fro",
        mask=mask,
        batch_size=8,
        extra_output=ExtraOutput.RANK,
    )

    assert rank_map is not None
    assert rank_map.shape == data.shape
    assert np.all(rank_map[mask] >= 0)
    assert np.any(rank_map[mask] > 0)


def test_main_gpu_float64_input(noisy_phantom_4d):
    """float64 input (e.g. from nibabel's get_fdata()) must denoise correctly.

    Regression test: the Triton accumulation kernel reinterprets its output
    buffer as raw float32 regardless of the actual dtype, so float64 input
    used to be silently read back as garbage.
    """
    from patch_denoise.gpu.main import main_gpu

    phantom, real, _ = noisy_phantom_4d
    data64 = real.astype(np.float64)
    mask = np.ones(data64.shape[:-1], dtype=bool)

    denoised, _, _, _, _ = main_gpu(
        data64,
        patch_shape=PATCH_SHAPE,
        patch_overlap=PATCH_OVERLAP,
        mask_threshold=50,
        recombination="weighted",
        method="optimal-fro",
        mask=mask,
        batch_size=8,
    )

    assert denoised.shape == data64.shape
    assert np.all(np.isfinite(denoised))
    noise_before = np.sqrt(np.nanmean(np.nanvar(data64 - phantom, axis=-1)))
    noise_after = np.sqrt(np.nanmean(np.nanvar(denoised - phantom, axis=-1)))
    assert noise_after < noise_before


@pytest.mark.parametrize("method", GPU_METHODS)
def test_main_gpu_center_full_time_no_nan(noisy_phantom_4d, method):
    """'center' recombination with a full-extent time axis must not produce NaN.

    PATCH_SHAPE's time axis (8) already equals the phantom's full time
    extent. Regression test for main_gpu unconditionally dividing the
    accumulator by the weight map (0/0 -> NaN for every voxel that isn't
    the single collapsed time point) instead of keeping the whole time
    profile at each spatial center like the CPU backend.
    """
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)
    # Maximal overlap on the spatial axes; the time axis is exempt since
    # it already spans the full data extent.
    patch_overlap = (5, 5, 5, 0)

    cpu_kwargs = dict(
        patch_shape=PATCH_SHAPE,
        patch_overlap=patch_overlap,
        mask=mask,
        recombination="center",
    )
    if method == "mp-pca":
        cpu_denoised, *_ = mp_pca(data, **cpu_kwargs, **_method_kwargs(method))
    else:
        cpu_denoised, *_ = optimal_thresholding(
            data, loss=method.split("-")[-1], **cpu_kwargs
        )
    assert np.all(np.isfinite(cpu_denoised))

    gpu_denoised, _, _, _, _ = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=patch_overlap,
        mask_threshold=50,
        recombination="center",
        method=method,
        mask=mask,
        batch_size=8,
        **_method_kwargs(method),
    )
    assert np.all(np.isfinite(gpu_denoised))
    npt.assert_allclose(gpu_denoised, cpu_denoised, rtol=1e-5, atol=1e-2)


def test_main_gpu_center_uneven_batch_extra_output(noisy_phantom_4d):
    """'center' recombination with COUNT/NOISE_STD/RANK extras, uneven batch.

    Regression guard: the "center" branch wrote extra-output values through
    a fixed-``batch_size``-length ``ones_buf`` (the COUNT accumulator)
    directly into per-patch output slices sized from the actual (possibly
    smaller) last batch; a batch_size that doesn't divide the patch count
    used to raise a shape-mismatch RuntimeError there. Also guards the
    NOISE_STD/RANK maps against reintroducing NaN outside the mask.
    """
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)
    # Maximal overlap on the spatial axes, as 'center' recombination requires;
    # the time axis is exempt since it already spans the full data extent.
    patch_overlap = (5, 5, 5, 0)

    _, _, noise_std, rank_map, counts = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=patch_overlap,
        mask_threshold=50,
        recombination="center",
        method="optimal-fro",
        mask=mask,
        batch_size=8,  # 6859 selected patches here, does not divide evenly by 8
        extra_output=ExtraOutput.NOISE_STD | ExtraOutput.RANK | ExtraOutput.COUNT,
    )

    assert np.all(np.isfinite(noise_std))
    assert np.all(np.isfinite(rank_map))
    assert np.all(np.isfinite(counts))


@pytest.mark.parametrize("method", GPU_METHODS)
def test_main_gpu_center_requires_maximal_overlap(noisy_phantom_4d, method):
    """'center' recombination must reject non-maximal overlap, CPU and GPU.

    'center' only keeps, per patch, the single voxel at its center. With
    less-than-maximal overlap (as PATCH_OVERLAP is here: 3, not
    PATCH_SHAPE-1=5), patch centers form a sparse lattice, so most voxels
    are never any patch's center and would be silently left undenoised.
    Both backends must reject this configuration outright.
    """
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    mask = np.ones(data.shape[:-1], dtype=bool)

    if method == "mp-pca":
        with pytest.raises(ValueError, match="maximal overlap"):
            mp_pca(
                data,
                patch_shape=PATCH_SHAPE,
                patch_overlap=PATCH_OVERLAP,
                mask=mask,
                recombination="center",
                threshold_scale=2.3,
            )
    else:
        with pytest.raises(ValueError, match="maximal overlap"):
            optimal_thresholding(
                data,
                patch_shape=PATCH_SHAPE,
                patch_overlap=PATCH_OVERLAP,
                mask=mask,
                recombination="center",
                loss=method.split("-")[-1],
            )

    with pytest.raises(ValueError, match="maximal overlap"):
        main_gpu(
            data,
            patch_shape=PATCH_SHAPE,
            patch_overlap=PATCH_OVERLAP,
            mask_threshold=50,
            recombination="center",
            method=method,
            mask=mask,
            batch_size=5,
            **_method_kwargs(method),
        )
