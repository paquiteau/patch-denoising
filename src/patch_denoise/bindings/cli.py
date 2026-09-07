#!/usr/bin/env python3
"""Cli interface."""

import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from nilearn.maskers import NiftiMasker

import cyclopts.validators as validators
import numpy as np
from cyclopts import App, Parameter, Token
from cyclopts.types import ExistingFile
from numpy.typing import NDArray
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from patch_denoise.bindings.utils import (
    DENOISER_MAP,
    fast_cuda_check,
    load_as_array,
    load_complex_nifti,
    save_array,
)
from patch_denoise.space_time.base import (
    DenoiserName,
    ExtraOutput,
    Recombination,
)

GPU_AVAILABLE = fast_cuda_check()

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[
        RichHandler(show_time=False, show_path=True, show_level=True, markup=True)
    ],
)
logging.captureWarnings(True)


def parse_dims(_, tokens: Sequence[Token]) -> tuple[int, ...]:
    """Parse a string representing dimensions into a 3 or 4-tuple of integers."""
    value = tokens[0].value
    dims = [int(x) for x in re.findall(r"-?\d+", value)]
    if len(dims) in (1, 3, 4):
        return tuple(dims)

    raise ValueError(
        "Must be an int, 3-tuple, or 4-tuple ('11' or '11x11x11')"
        " any 1-character separator is allowed (except space and -): "
        "('11x11x11', 11_11_11', '11,11,11')"
    )


def bids_extra_path(output_file: Path, name: str) -> Path:
    """Insert a BIDS-like ``_<name>`` suffix before the NIfTI extension."""
    for ext in (".nii.gz", ".nii"):
        if output_file.name.endswith(ext):
            stem = output_file.name[: -len(ext)]
            return output_file.with_name(f"{stem}_{name}{ext}")
    return output_file.with_name(f"{output_file.stem}_{name}{output_file.suffix}")


app = App(
    help="Patch denoising CLI tool.", default_parameter=Parameter(short_alias=True)
)


_PATCH_PARAM_HELP = (
    'Dimension separated by ",". Use -1 to use the entire dimension. '
    "Example: 11,11,11,-1."
)

MethodOpt = Annotated[DenoiserName, Parameter(help="Denoising method.")]
PatchShapeOpt = Annotated[
    tuple[int, ...],
    Parameter(
        alias="-ps",
        negative="",
        converter=parse_dims,
        help="Patch shape. " + _PATCH_PARAM_HELP,
    ),
]
PatchOverlapOpt = Annotated[
    tuple[int, ...],
    Parameter(
        alias="-po",
        negative="",
        converter=parse_dims,
        help="Patch overlap. " + _PATCH_PARAM_HELP,
    ),
]

RecombinationOpt = Annotated[
    Recombination,
    Parameter(name=("--recombination", "-r"), help="Recombination method."),
]
MaskOpt = Annotated[
    ExistingFile | None,
    Parameter(
        name=("--mask", "-k"),
        help="Mask NIfTI file (3D). if None, mask is computed automatically.",
    ),
]
MaskThreshOpt = Annotated[
    int,
    Parameter(
        name=("--mask-threshold", "-t"),
        help="Min % of overlap between a patch and the mask to trigger computation.",
    ),
]
ExtraOpts = Annotated[
    dict[str, float | str] | None,
    Parameter(alias="-e", help="Extra parameters for the denoiser as -e.key=value"),
]

NaN2NumOpt = Annotated[
    float | None, Parameter(help="Replace any NaN in input-data with VALUE")
]

VerboseOpt = Annotated[
    int, Parameter(count=True, help="Repeat for increase verbosity (-vvv)")
]

GpuFlag = Annotated[
    bool,
    Parameter(
        name="--gpu",
        negative="--cpu",
        help="Use GPU or CPU for computation. Requires patch_denoise.gpu module. "
        "GPU is enabled by default if available",
    ),
]
GpuBatchSizeOpt = Annotated[
    int,
    Parameter(
        "--gpu-batch-size",
        help="Number of patches processed per GPU batch. If 0, measure "
        "the fastest size for this GPU/method/patch-shape once and cache it "
        "(~/.cache/patch_denoise/gpu_batch_size.json) for future runs. "
        "Ignored on CPU.",
    ),
]
OutMapExtraOpt = Annotated[
    ExtraOutput,
    Parameter(
        "--outmap",
        negative="",
        consume_multiple=True,
    ),
]
_NO_EXTRA_OUTPUT = ExtraOutput(0)


#############
# Main CLI  #
#############


def _load_noise_std(
    noise_std_map_file: Path | None,
    noise_std_map_phase_file: Path | None,
) -> tuple[NDArray | None, NDArray | None]:
    if noise_std_map_file is not None and noise_std_map_phase_file is not None:
        noise_std_map, affine_noise_map = load_complex_nifti(
            noise_std_map_file,
            noise_std_map_phase_file,
        )
    elif noise_std_map_file is not None:
        noise_std_map, affine_noise_map = load_as_array(noise_std_map_file)
    elif noise_std_map_phase_file is not None:
        raise ValueError(
            "The phase component of the noise map has been provided, "
            "but not the magnitude."
        )
    else:
        noise_std_map = None
        affine_noise_map = None
    return noise_std_map, affine_noise_map


def _load_validate_input(
    input_file: Path,
    input_phase: Path | None,
    mask: Path | None,
    noise_std_map_file: Path | None,
    noise_std_map_phase_file: Path | None,
    nan_to_num: float | None,
    verbose: int,
) -> tuple[NDArray, NDArray, "NiftiMasker", NDArray | None]:
    from nilearn.image import resample_img
    from nilearn.maskers import NiftiMasker

    if input_phase is not None:
        input_data, affine = load_complex_nifti(input_file, input_phase)
    else:
        input_data, affine = load_as_array(input_file)

    log.info(f"Input data shape: {input_data.shape}")
    n_nans = np.isnan(input_data).sum()
    if n_nans > 0:
        log.warning(
            f"{n_nans}/{input_data.size} voxels are NaN. "
            "You might want to use --nan-to-num=<value>",
            stacklevel=0,
        )

    if nan_to_num is not None:
        input_data = np.nan_to_num(input_data, nan=nan_to_num)

    masker = NiftiMasker(verbose=verbose, mask_strategy="epi")
    if mask is not None:
        masker.mask_img = mask
        masker.fit()
    else:
        masker.fit(input_file)

    affine_mask = masker.mask_img_.affine

    noise_std_map, affine_noise = _load_noise_std(
        noise_std_map_file, noise_std_map_phase_file
    )

    if affine is not None:
        if (affine_mask is not None) and not np.allclose(affine, affine_mask):
            log.warning(
                "Affine matrix of input and mask does not match, it will be resampled",
                stacklevel=2,
            )

            masker.mask_img_ = resample_img(
                masker.mask_img_,
                target_affine=affine,
                target_shape=input_data.shape[:3],
                interpolation="nearest",
            )

        if (affine_noise is not None) and not np.allclose(affine, affine_noise):
            log.warning(
                "Affine matrix of input and noise map does not match", stacklevel=2
            )

    return input_data, affine, masker, noise_std_map


@app.default()
def main(
    input_file: ExistingFile,
    output_file: Annotated[
        Path | None,
        Parameter(
            validator=validators.Path(dir_okay=False),
            help="Output denoised NIfTI file (4D). Default is D<input_file>.",
        ),
    ] = None,
    *,
    method: MethodOpt = DenoiserName.OPTIMAL_FRO,
    patch_shape: PatchShapeOpt = (11, 11, 11, -1),
    patch_overlap: PatchOverlapOpt = (5, 5, 5, -1),
    recombination: RecombinationOpt = Recombination.WEIGHTED,
    mask: MaskOpt = None,
    mask_threshold: MaskThreshOpt = 50,
    extras: ExtraOpts = None,
    nan_to_num: NaN2NumOpt = 0.0,
    verbose: VerboseOpt = 0,
    gpu: GpuFlag = GPU_AVAILABLE,
    gpu_batch_size: GpuBatchSizeOpt = 0,
    outmap_extra: OutMapExtraOpt = _NO_EXTRA_OUTPUT,
    input_phase: Annotated[
        ExistingFile | None,
        Parameter(
            alias="-ip",
            help="Input phase NIfTI file (4D). If provided, process complex data.",
        ),
    ] = None,
    noise_std_map: Annotated[
        ExistingFile | None,
        Parameter(help="Input Noise std map"),
    ] = None,
    noise_std_map_phase: Annotated[
        ExistingFile | None,
        Parameter(help="Input Noise std map, phase component."),
    ] = None,
):
    """Perform local-low-rank denoising on 4D MRI data."""
    tic0 = tic = time.perf_counter()
    kwargs: dict[str, Any] = dict(extras or {})

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose, len(levels) - 1)]
    logging.getLogger("patch_denoise").setLevel(level)
    logging.getLogger("py.warnings").setLevel(level)
    if output_file is None:
        output_file = input_file.parent / f"D{input_file.name}"

    parent_dir = output_file.parent
    if not output_file.parent.exists():
        parent_dir.mkdir(exist_ok=True, parents=True)
        log.info(f"{output_file.parent} created")
    if output_file.exists():
        log.warning(f"{output_file} will be overwritten")

    if ExtraOutput.COUNT in outmap_extra and not gpu:
        raise ValueError("--outmap.count requires --gpu")

    extra_output_files = {
        n: bids_extra_path(output_file, n.name.lower()) for n in outmap_extra
    }
    for extra_file in extra_output_files.values():
        if not extra_file.parent.exists():
            extra_file.parent.mkdir(exist_ok=True, parents=True)
            log.info(f"{extra_file.parent} created")
        if extra_file.exists():
            log.warning(f"{extra_file} will be overwritten")

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task(description="Loading and validating input...", total=None)
        # 2. Add your task and start the progress display
        input_data, affine, masker, noise_std_map_data = _load_validate_input(
            input_file,
            input_phase,
            mask,
            noise_std_map,
            noise_std_map_phase,
            nan_to_num,
            verbose,
        )
    if mask is None:
        mask_filename = output_file.with_stem("mask_" + output_file.stem)
        log.info("Saving automatically computed mask to {mask_filename}.")
        masker.mask_img_.to_filename(mask_filename)
        log.info("Creating report for NiftiMasker.")
        report = masker.generate_report()
        report.save_as_html(output_file.with_suffix(".html"))
    mask_data = masker.mask_img_.get_fdata().astype(bool)

    toc = time.perf_counter()
    from patch_denoise.space_time.base import _patch_param

    # substitute any -1 in patch_shape or patch_overlap with the corresponding dimension
    # of input_data
    patch_shape_ = _patch_param(patch_shape, input_data.shape)

    patch_overlap_ = _patch_param(patch_overlap, input_data.shape)
    log.debug("Preprocessing and data loading completed in %.2f seconds.", toc - tic)
    log.info(f"denoising method: {method}.")
    log.info(f"patch shape: {patch_shape_} (from {patch_shape}).")
    log.info(f"patch overlap: {patch_overlap_} (from {patch_overlap}).")
    log.info(f"recombination method: {recombination}.")
    log.info(f"mask threshold: {mask_threshold}.")
    log.info(f"GPU: {gpu}.")
    log.info(f"extra parameters: {kwargs}.")
    log.info(f"nan_to_num: {nan_to_num}.")
    log.info(f"input data shape: {input_data.shape}.")
    log.info(msg=f"mask shape: {masker.mask_img_.shape}.")
    noise_std_map_shape = (
        noise_std_map_data.shape if noise_std_map_data is not None else None
    )
    log.info(f"noise std map: {noise_std_map_shape}.")
    log.info(f"output file: {output_file}.")
    log.info(f"extra outputs: {outmap_extra or 'none'}.")
    log.debug(f"extra output files: {list(extra_output_files.values())}.")
    log.debug(f"input affine:\n{affine}.")
    log.debug(f"mask affine: \n{masker.mask_img_.affine}.")

    if gpu:
        from patch_denoise.gpu.main import GPU_SUPPORTED_METHODS
        from patch_denoise.gpu.main import main_gpu as denoise_func

        if method not in GPU_SUPPORTED_METHODS:
            raise ValueError(f"Method {method} is not supported on GPU. ")
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "GPU support is not available. Please ensure that the "
                "patch_denoise.gpu module is installed and that you have"
                "a compatible GPU."
            )
        log.info("Using GPU for computation.")

        kwargs["method"] = method
        kwargs["batch_size"] = gpu_batch_size
        # Only accumulate/return the extras actually requested for saving.
        kwargs["extra_output"] = outmap_extra
    else:
        denoise_func = DENOISER_MAP[method]

    if method in [
        DenoiserName.NORDIC,
        DenoiserName.HYBRID_PCA,
        DenoiserName.ADAPTIVE_QUT,
        DenoiserName.OPTIMAL_FRO_NOISE,
    ]:
        if noise_std_map is None:
            raise RuntimeError("A noise map must be specified for this method.")
        kwargs["noise_std"] = noise_std_map_data

    tic = time.perf_counter()
    result = denoise_func(
        input_data,
        patch_shape=patch_shape_,
        patch_overlap=patch_overlap_,
        mask=mask_data,
        mask_threshold=mask_threshold,
        recombination=recombination,
        **kwargs,
    )
    # GPU returns a 5-tuple (..., count_map), CPU a 4-tuple (no count output).
    denoised_data, weights_map, var_map, rank_map, *extra_tail = result
    count_map = extra_tail[0] if extra_tail else None
    toc = time.perf_counter()
    log.debug("Denoising completed in %.2f seconds.", toc - tic)
    tic = time.perf_counter()
    save_array(denoised_data, affine, output_file)
    extra_arrays = {
        ExtraOutput.WEIGHTS: weights_map,
        ExtraOutput.NOISE_STD: var_map,
        ExtraOutput.RANK: rank_map,
        ExtraOutput.COUNT: count_map,
    }
    for name, extra_file in extra_output_files.items():
        array = extra_arrays[name]
        assert array is not None, f"{name} was requested but not returned"
        save_array(array, affine, extra_file)
    toc = time.perf_counter()
    log.debug("Saving completed in %.2f seconds.", toc - tic)
    log.debug("Total time: %.2f seconds.", toc - tic0)


if __name__ == "__main__":
    app()
