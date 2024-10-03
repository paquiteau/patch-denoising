#!/usr/bin/env python3
"""Cli interface."""

import argparse
from pathlib import Path
import logging

import numpy as np

from .utils import (
    DENOISER_MAP,
    DenoiseParameters,
    compute_mask,
    load_as_array,
    save_array,
    load_complex_nifti,
)
from patch_denoise import __version__


DENOISER_NAMES = ", ".join(d for d in DENOISER_MAP if d)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help="Input (noisy) file.", type=Path)
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        type=Path,
        help=("Output (denoised) file.\n" "default is D<input_file_name>"),
    )

    parser.add_argument("--version", action="version", version=__version__)

    denoising_group = parser.add_argument_group("Denoising parameters")

    conf_vs_separate = denoising_group.add_mutually_exclusive_group(required=True)
    conf_vs_separate.add_argument(
        "--method",
        help=(
            "Denoising method.\n"
            f"Available denoising methods:\n  {DENOISER_NAMES}\n"
            "This parameter is mutually exclusive with --conf."
        ),
        choices=DENOISER_MAP,
        default="optimal-fro",
    )

    denoising_group.add_argument(
        "--patch-size",
        help=(
            "Patch size. \n"
            "If int, this is the size of the patch in each dimension. \n"
            "If not specified, the default value is used. \n"
            "Note: setting a low aspect ratio will increase the number of patch to be"
            "processed, and will increase memory usage and computation times. "
            "This parameter should be used in conjunction with --method and "
            "is mutually exclusive with --conf."
        ),
        default=11,
        type=int,
    )
    denoising_group.add_argument(
        "--patch-overlap",
        help=(
            "Patch overlap. \n"
            "If int, this is the size of the overlap in each dimension. \n"
            "If not specified, the default value is used. \n"
            "Note: setting a low overlap will increase the number of patch to be "
            "processed, and will increase memory usage and computation times. "
            "This parameter should be used in conjunction with --method and "
            "is mutually exclusive with --conf."
        ),
        default=5,
        type=int,
    )
    denoising_group.add_argument(
        "--recombination",
        help=(
            "Recombination method. \n"
            "If 'mean', the mean of the overlapping patches is used. \n"
            "If 'weighted', the weighted mean of the overlapping patches is used. \n"
            "If not specified, the default value is used. "
            "This parameter should be used in conjunction with --method and "
            "is mutually exclusive with --conf."
        ),
        default="weighted",
        choices=["mean", "weighted"],
    )
    denoising_group.add_argument(
        "--extra",
        metavar="key=value",
        default=None,
        nargs="*",
        help="extra key=value arguments for denoising methods.",
    )
    conf_vs_separate.add_argument(
        "--conf",
        help=(
            "Denoising configuration.\n"
            "Format should be <name>_<patch-size>_<patch-overlap>_<recombination>.\n"
            "See Documentation of 'DenoiseParameter.from_str' for full specification.\n"
            f"Available denoising methods:\n  {DENOISER_NAMES}\n"
            "This parameter is mutually exclusive with --method."
        ),
        default=None,
    )

    data_group = parser.add_argument_group("Additional input data")
    data_group.add_argument(
        "--mask",
        metavar="FILE|auto",
        default=None,
        help=(
            "mask file, if auto or not provided"
            " it would be determined from the average image."
        ),
    )
    data_group.add_argument(
        "--noise-map",
        metavar="FILE",
        default=None,
        type=Path,
        help="noise map estimation file",
    )
    data_group.add_argument(
        "--input-phase",
        metavar="FILE",
        default=None,
        type=Path,
        help=(
            "Phase of the input data. This MUST be in radian. "
            "No conversion would be applied."
        ),
    )

    misc_group = parser.add_argument_group("Miscellaneous options")
    misc_group.add_argument(
        "--time-slice",
        help=(
            "Slice across time. \n"
            "If <N>x the patch will be N times longer in space than in time \n"
            "If int, this is the size of the time dimension patch. \n"
            "If not specified, the whole time series is used. \n"
            "Note: setting a low aspect ratio will increase the number of patch to be"
            "processed, and will increase memory usage and computation times."
        ),
        default=None,
        type=str,
    )
    misc_group.add_argument(
        "--output-noise-map",
        metavar="FILE",
        default=None,
        type=Path,
        help="Output name for calculated noise map",
    )
    misc_group.add_argument(
        "--nan-to-num",
        metavar="VALUE",
        default=None,
        type=float,
        help="Replace NaN by the provided value.",
    )
    misc_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level. You can provide multiple times (e.g., -vvv).",
    )

    args = parser.parse_args()

    # default value for output.
    if args.output_file is None:
        args.output_file = args.input_file.with_stem("D" + args.input_file.stem)

    if args.extra:
        key_values = [kv.split("=") for kv in args.extra]
        args.extra = {}
        for k, v in key_values:
            try:
                v = float(v)
            except ValueError:
                pass
            args.extra[k] = v
    else:
        args.extra = {}

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(args.verbose, len(levels) - 1)]  # cap to last level index
    logging.basicConfig(level=level)

    return args


def main():
    """Command line entry point."""
    args = parse_args()

    if args.input_phase is not None:
        input_data, affine = load_complex_nifti(args.input_file, args.input_phase)
    input_data, affine = load_as_array(args.input_file)

    if args.nan_to_num is not None:
        input_data = np.nan_to_num(input_data, nan=args.nan_to_num)
    n_nans = np.isnan(input_data).sum()
    if n_nans > 0:
        logging.warning(
            f"{n_nans}/{np.prod(input_data.shape)} voxels are NaN."
            " You might want to use --nan-to-num=<value>",
            stacklevel=0,
        )

    if args.mask == "auto":
        mask = compute_mask(input_data)
        affine_mask = None
    else:
        mask, affine_mask = load_as_array(args.mask)
    noise_map, affine_noise_map = load_as_array(args.noise_map)

    if affine is not None:
        if affine_mask is not None and np.allclose(affine, affine_mask):
            logging.warning(
                "Affine matrix of input and mask does not match", stacklevel=2
            )
        if affine_noise_map is not None and np.allclose(affine, affine_noise_map):
            logging.warning(
                "Affine matrix of input and noise map does not match", stacklevel=2
            )

    d_par = DenoiseParameters.from_str(args.conf)
    if isinstance(args.time_slice, str):
        if args.time_slice.endswith("x"):
            t = float(args.time_slice[:-1])
            t = int(d_par ** (input_data.ndim - 1) / t)
        else:
            t = int(args.time_slice)
        d_par.patch_shape = (d_par.patch_shape,) * (input_data.ndim - 1) + (t,)
    print(d_par)
    denoise_func = DENOISER_MAP[d_par.method]
    extra_kwargs = dict()

    if d_par.method in [
        "nordic",
        "hybrid-pca",
        "adaptive-qut",
        "optimal-fro-noise",
    ]:
        extra_kwargs["noise_std"] = noise_map
        if noise_map is None:
            raise RuntimeError("A noise map must be specified for this method.")

    denoised_data, patchs_weight, noise_std_map, rank_map = denoise_func(
        input_data,
        patch_shape=d_par.patch_shape,
        patch_overlap=d_par.patch_overlap,
        mask=mask,
        mask_threshold=d_par.mask_threshold,
        recombination=d_par.recombination,
        **extra_kwargs,
        **args.extra,
    )

    save_array(denoised_data, affine, args.output_file)
    save_array(noise_std_map, affine, args.output_noise_map)


if __name__ == "__main__":
    main()
