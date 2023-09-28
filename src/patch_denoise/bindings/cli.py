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


DENOISER_NAMES = ", ".join(d for d in DENOISER_MAP if d)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_file", help="Input (noisy) file.", type=Path)
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        type=Path,
        help=("Output (denoised) file.\n" "default is D<input_file_name>"),
    )

    parser.add_argument(
        "--conf",
        help=(
            "Denoising configuration.\n"
            "Format should be <name>_<patch-size>_<patch-overlap>_<recombination>.\n"
            "See Documentation of 'DenoiseParameter.from_str' for full specification.\n"
            "Default:  optimal-fro_11_5_weighted\n"
            "Available denoising methods:\n  " + DENOISER_NAMES
        ),
        default="optimal-fro_11_5_weighted",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help=(
            "mask file, if auto or not provided"
            " it would be determined from the average image."
        ),
    )
    parser.add_argument(
        "--noise-map",
        default=None,
        help="noise map estimation file",
    )
    parser.add_argument(
        "--output-noise-map",
        default=None,
        help="output noise map estimation file",
    )
    parser.add_argument(
        "--extra",
        default=None,
        nargs="*",
        help="extra key=value arguments for denoising methods.",
    )
    parser.add_argument(
        "--nan-to-num",
        default=None,
        type=float,
        help="Replace NaN by the provided value.",
    )
    parser.add_argument(
        "--input-phase",
        default=None,
        type=Path,
        help=(
            "Phase of the input data. This MUST be in radian. "
            "No conversion would be applied."
        ),
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)

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
            raise RuntimeError("A noise map must me specified for this method.")
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
