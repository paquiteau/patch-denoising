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
# Maximal overlap, as 'center' recombination requires.
CENTER_OVERLAP = (5, 5, 5, 0)
GPU_METHODS = ["mp-pca", "optimal-fro", "optimal-nuc", "optimal-ope"]


def _method_kwargs(method):
    """Extra denoising kwargs that only apply to a given method."""
    return {"threshold_scale": 0.9} if method == "mp-pca" else {}


@pytest.fixture(scope="module")
def noisy_phantom_4d(rng):
    """Create a small 4D (X, Y, Z, T) noisy phantom, real and imaginary parts."""
    phantom = add_frames(mr_shepp_logan_t2_star(24), 8)
    g_map = g_factor_map(phantom.shape[:-1])
    real = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    imag = add_temporal_gaussian_noise(phantom, sigma=1, rng=rng, g_factor_map=g_map)
    return phantom, real, imag


def _gpu_and_cpu(data, patch_shape, patch_overlap, method, recombination):
    """Run ``main_gpu`` and the CPU reference.

    Return both ``(denoised, noise std, rank)``.
    """
    from patch_denoise.gpu.main import main_gpu

    kwargs = dict(
        patch_shape=patch_shape,
        patch_overlap=patch_overlap,
        mask=None,
        recombination=recombination,
    )
    if method == "mp-pca":
        cpu_denoised, _, cpu_noise_std, _ = mp_pca(
            data,
            patch_shape=PATCH_SHAPE,
            patch_overlap=PATCH_OVERLAP,
            mask=mask,
            recombination="weighted",
            threshold_scale=0.9,
        )
    else:
        cpu_denoised, _, cpu_noise_std, cpu_rank = optimal_thresholding(
            data, **kwargs, loss=method.split("-")[-1]
        )
    gpu_denoised, _, gpu_noise_std, gpu_rank, _ = main_gpu(
        data,
        **kwargs,
        mask_threshold=50,
        method=method,
        batch_size=7,  # does not divide the patch counts
        extra_output=ExtraOutput.NOISE_STD | ExtraOutput.RANK,
        **_method_kwargs(method),
    )
    gpu = (gpu_denoised, gpu_noise_std, gpu_rank)
    cpu = (cpu_denoised, cpu_noise_std, cpu_rank)
    for g, c in zip(gpu, cpu):
        assert g.shape == c.shape == data.shape
    return gpu, cpu


@pytest.mark.parametrize("recombination", ["weighted", "average"])
@pytest.mark.parametrize("method", GPU_METHODS)
def test_main_gpu_matches_cpu(noisy_phantom_4d, method, recombination):
    """GPU denoising, noise std and rank maps must match the CPU backend.

    Uses a batch size that does not divide the number of patches, so the
    last, smaller batch goes through a second eigensolver workspace.
    """
    _, real, _ = noisy_phantom_4d
    data = real.astype(np.float32)
    gpu, cpu = _gpu_and_cpu(data, PATCH_SHAPE, PATCH_OVERLAP, method, recombination)
    assert np.all(gpu[0] != 0)  # every voxel is in a patch
    for g, c in zip(gpu, cpu):
        npt.assert_allclose(g, c, rtol=1e-5, atol=1e-2)


def _layout(noisy_phantom_4d, layout, center=False):
    """Return the data, patch shape and overlap of a test layout."""
    _, real, imag = noisy_phantom_4d
    data = real.astype(np.float32)
    patch_shape = PATCH_SHAPE
    patch_overlap = CENTER_OVERLAP if center else PATCH_OVERLAP
    if layout == "time-window":
        # T = 16; independent noise in both halves (a mirrored time axis
        # makes near-degenerate patches, whose rank flips on rounding).
        data = np.concatenate([data, imag.astype(np.float32)], axis=-1)
        patch_overlap = (*patch_overlap[:-1], 7 if center else 4)
    elif layout == "2d+t":
        data = data[:, :, data.shape[2] // 2]
        patch_shape = (*PATCH_SHAPE[:2], PATCH_SHAPE[-1])
        patch_overlap = (*patch_overlap[:2], patch_overlap[-1])
    elif layout == "fortran-order":
        data = np.asfortranarray(data)
    elif layout == "ragged-stride":
        # A stride of 4 leaves a remainder on 24-voxel axes, covered by a
        # last patch aligned on the border.
        patch_overlap = (2, 2, 2, 0)
    elif layout == "float64":
        data = real.astype(np.float64)
    elif layout == "complex64":
        data = (real + 1j * imag).astype(np.complex64)
    return data, patch_shape, patch_overlap


@pytest.mark.parametrize(
    "layout",
    ["time-window", "2d+t", "fortran-order", "ragged-stride", "float64", "complex64"],
)
def test_main_gpu_layouts_match_cpu(noisy_phantom_4d, layout):
    """Reading/scattering patches in place must match the CPU backend.

    Covers what the in-place addressing depends on: patches shorter than the
    time axis (non-zero time offset, and a Gram matrix centered by projection
    rather than by the shift of the volume), the size-1 Z axis injected for
    2D+T input, non C-ordered input (e.g. from nibabel), a stride leaving a
    remainder, and the input dtypes.
    """
    data, patch_shape, patch_overlap = _layout(noisy_phantom_4d, layout)
    gpu, cpu = _gpu_and_cpu(data, patch_shape, patch_overlap, "optimal-fro", "weighted")
    assert np.all(gpu[0] != 0)
    for g, c in zip(gpu, cpu):
        npt.assert_allclose(g, c, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("layout", ["full-time", "time-window", "2d+t"])
def test_main_gpu_center_layouts_match_cpu(noisy_phantom_4d, layout):
    """'center' recombination from patches read in place must match the CPU.

    Without a mask, the data is padded so that border voxels are patch
    centers too: every voxel is denoised. A time axis spanning the whole
    data extent keeps the whole time profile at each spatial center.
    """
    data, patch_shape, patch_overlap = _layout(noisy_phantom_4d, layout, center=True)
    gpu, cpu = _gpu_and_cpu(data, patch_shape, patch_overlap, "optimal-fro", "center")
    # The noise std and rank maps differ: the CPU averages them over every
    # patch, the GPU only writes them at the patch centers.
    assert np.all(np.isfinite(gpu[0])) and np.all(gpu[0] != 0)
    npt.assert_allclose(gpu[0], cpu[0], rtol=1e-5, atol=1e-2)


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

    _, _, noise_std, rank_map, counts = main_gpu(
        data,
        patch_shape=PATCH_SHAPE,
        patch_overlap=CENTER_OVERLAP,
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


def test_main_gpu_center_requires_maximal_overlap(noisy_phantom_4d):
    """'center' recombination must reject non-maximal overlap, as on CPU."""
    from patch_denoise.gpu.main import main_gpu

    _, real, _ = noisy_phantom_4d
<<<<<<< variant A
>>>>>>> variant B
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
                threshold_scale=0.9,
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

======= end
    with pytest.raises(ValueError, match="maximal overlap"):
        main_gpu(
            real.astype(np.float32),
            patch_shape=PATCH_SHAPE,
            patch_overlap=PATCH_OVERLAP,
            mask_threshold=50,
            recombination="center",
            method="optimal-fro",
            mask=None,
            batch_size=5,
        )


def test_host_stager_matches_cpu():
    """Chunked copies through the pinned buffers must be exact, for any dtype."""
    import torch

    from patch_denoise.gpu.main import _STAGING_BYTES, _HostStager

    stager = _HostStager()
    n = 5 * (_STAGING_BYTES // 10 + 12345)  # several chunks, ragged last one
    for t in (
        torch.randn(n, device="cuda").reshape(-1, 5),
        torch.randn(n, device="cuda", dtype=torch.complex64),
        torch.randint(0, 100, (n,), device="cuda", dtype=torch.int32),
        torch.randn(7, 3, device="cuda").mT,  # non-contiguous
    ):
        out = stager(t)
        expected = t.cpu().numpy()
        assert out.dtype == expected.dtype
        npt.assert_array_equal(out, expected)
