#!/usr/bin/env python3
"""Cli interface."""

import json
import logging
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import typer
from nilearn.image import load_img
from nilearn.interfaces.bids import get_bids_files, parse_bids_filename
from nilearn.interfaces.bids.utils import bids_entities, create_bids_filename
from nilearn.maskers import NiftiMasker
from numpy.typing import NDArray
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from patch_denoise import __version__
from patch_denoise.bindings.utils import (
    DENOISER_MAP,
    fast_cuda_check,
    load_as_array,
    load_complex_nifti,
    save_array,
)

GPU_AVAILABLE = fast_cuda_check()

log = logging.getLogger("patch-denoise")

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logging.captureWarnings(True)

DENOISER_NAMES = ", ".join(d for d in DENOISER_MAP if d)


class AnalysisEnum(StrEnum):
    """Enum for BIDS analysis levels."""

    PARTICIPANT = "participant"


class DenoiserEnum(StrEnum):
    """Enum for denoising methods."""

    MP_PCA = "mp-pca"
    HYBRID_PCA = "hybrid-pca"
    RAW = "raw"
    OPTIMAL_FRO = "optimal-fro"
    OPTIMAL_FRO_NOISE = "optimal-fro-noise"
    OPTIMAL_NUC = "optimal-nuc"
    OPTIMAL_OPE = "optimal-ope"
    NORDIC = "nordic"
    ADAPTIVE_QUT = "adaptive-qut"


class RecombinationEnum(StrEnum):
    """Enum for recombination methods."""

    WEIGHTED = "weighted"
    MEAN = AVERAGE = "mean"


def parse_dims(value: Any) -> tuple[int, int, int, int]:
    """Parse a string representing dimensions into a 3 or 4-tuple of integers."""
    if not isinstance(value, str):
        return value  # Already a tuple of ints
    dims = [int(x) for x in re.findall(r"\d+", value)]
    if len(dims) == 1:
        return dims[0], dims[0], dims[0], -1
    if len(dims) == 3:
        return dims[0], dims[1], dims[2], -1
    if len(dims) == 4:
        return dims[0], dims[1], dims[2], dims[3]

    raise typer.BadParameter(
        "Must be an int, 3-tuple, or 4-tuple ('11' or '11x11x11')"
        " any 1-character separator is allowed "
        "('11x11x11', '11-11-11', '11_11_11', '11,11,11')"
    )


def parse_mask_arg(value: str):
    """Validate if value is 'auto' or points to an existing file."""
    if value == "auto":
        return value
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise typer.BadParameter(
            f"Path should point to a file, or be 'auto': <{value}>."
        )
    return path.absolute()


def parse_extra_args(extras: list[str] | None) -> dict[str, Any]:
    """Parse extra arguments passed as key=value pairs into a dictionary."""
    kwargs = dict()
    if extras is None:
        return kwargs
    for kv in extras:
        if "=" not in kv:
            raise typer.BadParameter(
                f"Extra parameter '{kv}' is not in key=value format."
            )
        key, value = kv.split("=", 1)
        try:
            value = float(value)
        except ValueError:
            pass  # keep as string if not a float
        kwargs[key] = value
    return kwargs


app = typer.Typer(help="Patch denoising CLI tool.")

###########################
## Shared Argument Types ##
###########################

MethodOpt = Annotated[
    DenoiserEnum,
    typer.Option(
        "-m", "--method", help=f"Denoising Method: Available: {DENOISER_NAMES}"
    ),
]
PatchShapeOpt = Annotated[
    tuple[int, int, int, int],
    typer.Option(
        "-ps",
        "--patch-shape",
        parser=parse_dims,
        metavar="X,Y,Z[,T]",
        help="Patch shape. If 4D a sliding window is used. "
        "If -1 is specified for a dimension, the entire dimension is put the patch.",
    ),
]
PatchOverlapOpt = Annotated[
    tuple[int, int, int, int],
    typer.Option(
        "-po",
        "--patch-overlap",
        parser=parse_dims,
        metavar="X,Y,Z[,T]",
        help="Patch overlap. If 4D a sliding window is used. "
        "If -1 is specified for a dimension, the entire dimension is put the patch.",
    ),
]

RecombinationOpt = Annotated[
    RecombinationEnum,
    typer.Option("-r", "--recombination", help="Recombination method."),
]
MaskOpt = Annotated[
    str,
    typer.Option(
        "-k",
        "--mask",
        callback=parse_mask_arg,
        help="Mask NIfTI file (3D). if auto, mask is computed automatically.",
        metavar="MASK_FILE | auto",
    ),
]
MaskThreshOpt = Annotated[
    int,
    typer.Option(
        "-t",
        "--mask-threshold",
        help="Min % of overlap between a patch and the mask to trigger computation.",
    ),
]
ExtraOpts = Annotated[
    list[str] | None,
    typer.Option(
        "-e",
        "--extra",
        help="Extra parameters for the denoising method, passed as key=value pairs. "
        "For example: --extra param1=val1 --extra param2=val2",
        metavar="KEY=VALUE",
    ),
]

NaN2NumOpt = Annotated[
    float | None, typer.Option(help="Replace any NaN in input-data with VALUE")
]
VerboseOpt = Annotated[
    int,
    typer.Option(
        "-v", "--verbose", count=True, help="Increase verbosity level (e.g., -vvv)."
    ),
]
GpuFlag = Annotated[
    bool,
    typer.Option(
        ...,
        "--gpu/--cpu",
        is_flag=True,
        help="Use GPU or CPU for computation. Requires patch_denoise.gpu module. "
        "GPU is enabled  by default if available.",
    ),
]


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
    mask: Path | str,
    noise_std_map_file: Path | None,
    noise_std_map_phase_file: Path | None,
    nan_to_num: float | None,
    verbose: int,
) -> tuple[NDArray, NDArray, NiftiMasker, NDArray | None]:
    if input_phase is not None:
        input_data, affine = load_complex_nifti(input_file, input_phase)
    else:
        input_data, affine = load_as_array(input_file)

    if nan_to_num is not None:
        input_data = np.nan_to_num(input_data, nan=nan_to_num)

    log.info(f"Input data shape: {input_data.shape}")
    n_nans = np.isnan(input_data).sum()
    if n_nans > 0:
        log.warning(
            f"{n_nans}/{input_data.size} voxels are NaN. "
            "You might want to use --nan-to-num=<value>",
            stacklevel=0,
        )

    masker = NiftiMasker(verbose=verbose, mask_strategy="epi")
    if mask != "auto":
        masker.mask_img = mask

    masker.fit(input_file)

    affine_mask = masker.mask_img_.affine

    noise_std_map, affine_noise = _load_noise_std(
        noise_std_map_file, noise_std_map_phase_file
    )

    if affine is not None:
        if (affine_mask is not None) and not np.allclose(affine, affine_mask):
            log.warning("Affine matrix of input and mask does not match", stacklevel=2)

        if (affine_noise is not None) and not np.allclose(affine, affine_noise):
            log.warning(
                "Affine matrix of input and noise map does not match", stacklevel=2
            )

    return input_data, affine, masker, noise_std_map


@app.command()
def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Input noisy NIfTI file (4D).",
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Argument(
            dir_okay=False,
            resolve_path=True,
            help="Output denoised NIfTI file (4D). Default is D<input_file>.",
        ),
    ] = None,
    output_noise_std_map_file: Annotated[
        Path | None,
        typer.Option(
            "--output-noise-std-map",
            dir_okay=False,
            resolve_path=True,
            help="Output noise level estimation NIfTI file (3D).",
        ),
    ] = None,
    method: MethodOpt = DenoiserEnum.OPTIMAL_FRO,
    patch_shape: PatchShapeOpt = (11, 11, 11, -1),
    patch_overlap: PatchOverlapOpt = (5, 5, 5, -1),
    recombination: RecombinationOpt = RecombinationEnum.WEIGHTED,
    mask: MaskOpt = "auto",
    mask_threshold: MaskThreshOpt = 50,
    extras: ExtraOpts = None,
    nan_to_num: NaN2NumOpt = None,
    verbose: VerboseOpt = 0,
    gpu: GpuFlag = GPU_AVAILABLE,
    input_phase: Annotated[
        Path | None,
        typer.Option(
            "-ip",
            "--input-phase",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Input phase NIfTI file (4D). If provided, process complex data.",
        ),
    ] = None,
    noise_std_map_file: Annotated[
        Path | None,
        typer.Option(
            "--noise-std-map",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Input Noise std map",
        ),
    ] = None,
    noise_std_map_phase_file: Annotated[
        Path | None,
        typer.Option(
            "--noise-std-map-phase",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Input Noise std map, phase component.",
        ),
    ] = None,
):
    """Perform local-low-rank denoising on 4D MRI data."""
    kwargs = parse_extra_args(extras)

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose, len(levels) - 1)]
    log.setLevel(level)
    logging.getLogger("py.warnings").setLevel(level)

    if output_file is None:
        output_file = input_file.parent / f"D{input_file.name}"

    parent_dir = Path(output_file).parent
    if not Path(output_file).parent.exists():
        parent_dir.mkdir(exist_ok=True, parents=True)
        log.info(f"{Path(output_file).parent} created")
    if not Path(output_file).exists():
        log.warning(f"{Path(output_file).parent} will be overwritten")

    # 1. Define only the columns you want (just the spinner and the text)
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task(description="Loading and validating input...", total=None)
        # 2. Add your task and start the progress display
        input_data, affine, masker, noise_std_map = _load_validate_input(
            input_file,
            input_phase,
            mask,
            noise_std_map_file,
            noise_std_map_phase_file,
            nan_to_num,
            verbose,
        )
    if mask == "auto":
        mask_filename = output_file.with_stem("mask_" + output_file.stem)
        log.info("Saving automatically computed mask to {mask_filename}.")
        masker.mask_img_.to_filename(mask_filename)
        log.info("Creating report for NiftiMasker.")
        report = masker.generate_report()
        report.save_as_html(output_file.with_suffix(".html"))
    mask_data = masker.mask_img_.get_fdata().astype(bool)

    # substitute any -1 in patch_shape or patch_overlap with the corresponding dimension
    # of input_data
    patch_shape = cast(
        tuple[int, int, int, int],
        tuple(p if p != -1 else s for p, s in zip(patch_shape, input_data.shape)),
    )
    patch_overlap = cast(
        tuple[int, int, int, int],
        tuple(p if p != -1 else s for p, s in zip(patch_overlap, input_data.shape)),
    )
    log.info(f"denoising method: {method}.")
    log.info(f"patch shape: {patch_shape}.")
    log.info(f"patch overlap: {patch_overlap}.")
    log.info(f"recombination method: {recombination}.")
    log.info(f"mask threshold: {mask_threshold}.")
    log.info(f"GPU: {gpu}.")
    log.info(f"extra parameters: {kwargs}.")
    log.info(f"nan_to_num: {nan_to_num}.")
    log.info(f"input data shape: {input_data.shape}.")
    log.info(msg=f"mask shape: {masker.mask_img_.shape}.")
    log.info(
        f"noise std map: {noise_std_map.shape if noise_std_map is not None else None}."
    )
    log.info(f"output file: {output_file}.")
    log.info(f"output noise std map file: {output_noise_std_map_file}.")
    log.info(f"input affine:\n{affine}.")
    log.info(f"mask affine: \n{masker.mask_img_.affine}.")

    if gpu:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "GPU support is not available. Please ensure that the "
                "patch_denoise.gpu module is installed and that you have"
                "a compatible GPU."
            )
        log.info("Using GPU for computation.")
        from patch_denoise.gpu.main import main_gpu

        denoised_data, _, noise_std_map = main_gpu(
            input_data, mask, noise_std_map, **kwargs
        )

    else:
        if method in [
            DenoiserEnum.NORDIC,
            DenoiserEnum.HYBRID_PCA,
            DenoiserEnum.ADAPTIVE_QUT,
            DenoiserEnum.OPTIMAL_FRO_NOISE,
        ]:
            if noise_std_map is None:
                raise RuntimeError("A noise map must be specified for this method.")
            kwargs["noise_std"] = noise_std_map
        denoise_func = DENOISER_MAP.get(method, None)
        if denoise_func is None:
            raise ValueError(f"Method {method} is not supported.")

        denoised_data, _, noise_std_map, _ = denoise_func(
            input_data,
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
            mask=mask_data,
            mask_threshold=mask_threshold,
            recombination=recombination,
            **kwargs,
        )

    save_array(denoised_data, affine, output_file)
    if output_noise_std_map_file is not None:
        save_array(noise_std_map, affine, output_noise_std_map_file)


############
# BIDS CLI #
############

bidsapp = typer.Typer(help="Patch denoising-bids CLI.")


@bidsapp.command()
def bids_main(
    bids_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Input BIDS directory.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Output BIDS directory.",
        ),
    ],
    analysis_level: Annotated[
        AnalysisEnum,
        typer.Argument(
            help="BIDS analysis level. Only 'participant' is supported.",
        ),
    ] = AnalysisEnum.PARTICIPANT,
    participant_label: Annotated[
        list[int] | None,
        typer.Option(
            "-participant-label",
            "--participant_label",
            help="List of participant labels to process. "
            "If not provided, all participants will be processed.",
        ),
    ] = None,
    session_label: Annotated[
        list[int] | None,
        typer.Option(
            "-session-label",
            "--session_label",
            help="List of session labels to process."
            " If not provided, all sessions will be processed.",
        ),
    ] = None,
    task_label: Annotated[
        list[str] | None,
        typer.Option(
            "-task-label",
            "--task_label",
            help="List of task labels to process."
            " If not provided, all tasks will be processed.",
        ),
    ] = None,
    bids_filters: Annotated[
        Path | None,
        typer.Option(
            "-bids-filters",
            "--bids_filters",
            help="Path to a JSON file containing BIDS filters. See "
            "https://fmriprep.org/en/latest/faq.html#"
            "how-do-i-select-only-certain-files-to-be-input-to-fmriprep",
        ),
    ] = None,
    method: MethodOpt = DenoiserEnum.OPTIMAL_FRO,
    patch_shape: PatchShapeOpt = (11, 11, 11, -1),
    patch_overlap: PatchOverlapOpt = (5, 5, 5, -1),
    recombination: RecombinationOpt = RecombinationEnum.WEIGHTED,
    mask: MaskOpt = "auto",
    mask_threshold: MaskThreshOpt = 50,
    extras: ExtraOpts = None,
    nan_to_num: NaN2NumOpt = None,
    verbose: VerboseOpt = 0,
    gpu: GpuFlag = GPU_AVAILABLE,
    noise_std_map_file: Annotated[
        Path | None,
        typer.Option(
            "--noise-std-map",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Input Noise std map",
        ),
    ] = None,
    noise_std_map_phase_file: Annotated[
        Path | None,
        typer.Option(
            "--noise-std-map-phase",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Input Noise std map, phase component.",
        ),
    ] = None,
):
    """Run CLI for bids app."""
    kwargs = parse_extra_args(extras)
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose, len(levels) - 1)]  # cap to last level index
    log.setLevel(level=level)

    if participant_label:
        all_subjects = participant_label
    else:
        all_subjects = [
            x.name.strip("sub-") for x in Path(bids_dir).iterdir() if "sub-" in x.name
        ]

    filters = []
    if task_label:
        filters.append(("task", task_label[0]))
    if session_label:
        filters.append(("ses", session_label[0]))

    noise_std_map, _ = _load_noise_std(noise_std_map_file, noise_std_map_phase_file)

    if gpu and not GPU_AVAILABLE:
        raise RuntimeError(
            "GPU support is not available. Please ensure that the "
            "patch_denoise.gpu module is installed and that you have"
            "a compatible GPU."
        )
    if method in [
        DenoiserEnum.NORDIC,
        DenoiserEnum.HYBRID_PCA,
        DenoiserEnum.ADAPTIVE_QUT,
        DenoiserEnum.OPTIMAL_FRO_NOISE,
    ]:
        if noise_std_map is None:
            raise RuntimeError("A noise map must be specified for this method.")
        kwargs["noise_std"] = noise_std_map
    denoise_func = DENOISER_MAP[method]

    output_dir.mkdir(exist_ok=True, parents=True)

    ds_json = output_dir / "dataset_description.json"
    GeneratedBy = {
        "Name": "patch-denoise",
        "Version": __version__,
        "Description": ("A dataset denoised with patch-denoise."),
        "CodeURL": "https://github.com/paquiteau/patch-denoising",
    }
    if ds_json.exists():
        with ds_json.open() as f_obj:
            dataset_description = json.load(f_obj)
        if dataset_description.get("GeneratedBy"):
            dataset_description["GeneratedBy"].append(GeneratedBy)
    else:
        dataset_description = {
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [GeneratedBy],
        }
    with ds_json.open("w") as f_obj:
        json.dump(dataset_description, f_obj, indent=4, sort_keys=True)

    entities_to_include = bids_entities()["raw"] + bids_entities()["derivatives"]

    for sub_label in all_subjects:
        files = get_bids_files(
            bids_dir,
            file_tag="bold",
            modality_folder="func",
            file_type="nii.*",
            sub_label=sub_label,
            filters=filters,
        )

        for f in files:
            parsed_f = parse_bids_filename(f)

            output_dir_for_subject = output_dir / f"sub-{parsed_f['entities']['sub']}"
            if "ses" in parsed_f["entities"]:
                output_dir_for_subject = (
                    output_dir_for_subject / f"ses-{parsed_f['entities']['ses']}"
                )

            parsed_f["entities"]["desc"] = "denoised"
            output_filename = output_dir_for_subject / create_bids_filename(
                parsed_f,
                entities_to_include=entities_to_include,
            )
            parsed_f["suffix"] = "mask"
            output_mask_filename = output_dir_for_subject / create_bids_filename(
                parsed_f,
                entities_to_include=entities_to_include,
            )
            parsed_f["suffix"] = "std"
            output_std_filename = output_dir_for_subject / create_bids_filename(
                parsed_f,
                entities_to_include=entities_to_include,
            )

            output_dir_for_subject.mkdir(exist_ok=True, parents=True)

            masker = NiftiMasker(verbose=verbose, mask_strategy="epi")
            masker.fit(f)
            mask = masker.mask_img_.get_fdata().astype(bool)
            affine = masker.mask_img_.affine

            report = masker.generate_report()
            report.save_as_html(output_mask_filename.with_suffix(".html"))

            masker.mask_img_.to_filename(output_mask_filename)

            input_data = load_img(f).get_fdata()

            if gpu:
                from patch_denoise.gpu.main import main_gpu

                denoised_data, _, noise_std_map = main_gpu(
                    input_data, mask, noise_std_map, **kwargs
                )
            else:
                denoised_data, _, noise_std_map, _ = denoise_func(
                    input_data,
                    patch_shape=patch_shape,
                    patch_overlap=patch_overlap,
                    mask=mask,
                    mask_threshold=mask_threshold,
                    recombination=recombination,
                    **kwargs,
                )

            print(noise_std_map.shape)
            save_array(denoised_data, affine, output_filename)
            save_array(noise_std_map, affine, output_std_filename)


if __name__ == "__main__":
    app()
