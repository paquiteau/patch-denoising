#!/usr/bin/env python3
"""Cli interface."""

import argparse
import re
import logging
from functools import partial
from pathlib import Path

import numpy as np

from patch_denoise.bindings.utils import (
    DENOISER_MAP,
    DenoiseParameters,
    compute_mask,
    load_as_array,
    save_array,
    load_complex_nifti,
)
from patch_denoise import __version__


DENOISER_NAMES = ", ".join(d for d in DENOISER_MAP if d)


def _path_exists(path, parser):
    """Ensure a given path exists."""
    if path is None or not Path(path).exists():
        raise parser.error(f"Path does not exist: <{path}>.")
    return Path(path).absolute()


def _is_file(path, parser):
    """Ensure a given path exists and it is a file."""
    path = _path_exists(path, parser)
    if not path.is_file():
        raise parser.error(
            f"Path should point to a file (or symlink of file): <{path}>."
        )
    return path


def _positive_int(string, is_parser=True):
    """Check if argument is an integer >= 0."""
    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        intarg = int(string)
    except ValueError:
        msg = "Argument must be a nonnegative integer."
        raise error(msg) from None

    if intarg < 0:
        raise error("Int argument must be nonnegative.")
    return intarg


def _tuple_positive_int(string, is_parser=True):
    """Parse patch size/overlap argument from string."""
    # find the separator (any non-digit character)
    sep = re.search(r"\D", string)
    if sep is not None:
        sep = sep.group(0)
        values = tuple(_positive_int(s, is_parser=is_parser) for s in string.split(sep))
    else:
        values = _positive_int(string, is_parser=is_parser)
    return values


class ToDict(argparse.Action):
    """A custom argparse "store" action to handle a list of key=value pairs."""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: U100
        """Call the argument."""
        d = {}
        for spec in values:
            try:
                key, value = spec.split("=")
            except ValueError:
                raise ValueError(
                    "Extra arguments must be in the form key=value."
                ) from None

            # Convert any float-like values to float
            try:
                value = float(value)
            except ValueError:
                pass

            d[key] = value
        setattr(namespace, self.dest, d)


def _get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)
    TuplePositiveInt = partial(_tuple_positive_int, is_parser=True)

    parser.add_argument(
        "input_file",
        help="Input (noisy) file.",
        type=IsFile,
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        type=Path,
        help=("Output (denoised) file.\nDefault is D<input_file_name>."),
    )

    parser.add_argument("--version", action="version", version=__version__)

    denoising_group = parser.add_argument_group("Denoising parameters")

    conf_vs_separate = denoising_group.add_mutually_exclusive_group(required=True)
    conf_vs_separate.add_argument(
        "--method",
        help=(
            "Denoising method.\n"
            f"Available denoising methods:\n  {DENOISER_NAMES}.\n"
            "This parameter is mutually exclusive with --conf."
        ),
        choices=DENOISER_MAP,
        default="optimal-fro",
    )

    denoising_group.add_argument(
        "--patch-shape",
        help=(
            "Patch shape.\n"
            "If int, this is the size of the patch in each dimension.\n"
            "If of format NxMx..., this is the size of the patch in each dimension\n"
            "Missing dimensions will be set to the size of the input data in that "
            "dimension.\n"
            "If not specified, the default value is used.\n"
            "Note: setting a low aspect ratio will increase the number of "
            "patches to be processed, "
            "and will increase memory usage and computation times.\n"
            "This parameter should be used in conjunction with --method and "
            "is mutually exclusive with --conf."
        ),
        default=11,
        type=TuplePositiveInt,
        metavar="INT",
    )
    denoising_group.add_argument(
        "--patch-overlap",
        help=(
            "Patch overlap.\n"
            "If int, this is the size of the overlap in each dimension.\n"
            "If not specified, the default value is used.\n"
            "Note: setting a low overlap will increase the number of patches "
            "to be processed, "
            "and will increase memory usage and computation times.\n"
            "This parameter should be used in conjunction with --method and "
            "is mutually exclusive with --conf."
        ),
        default=5,
        type=TuplePositiveInt,
        metavar="INT",
    )
    denoising_group.add_argument(
        "--recombination",
        help=(
            "Recombination method.\n"
            "If 'mean', the mean of the overlapping patches is used.\n"
            "If 'weighted', the weighted mean of the overlapping patches is used.\n"
            "This parameter should be used in conjunction with --method and "
            "is mutually exclusive with --conf."
        ),
        default="weighted",
        choices=["mean", "weighted"],
    )
    denoising_group.add_argument(
        "--mask-threshold",
        help=(
            "Mask threshold.\n"
            "If int, this is the threshold for the mask.\n"
            "If not specified, the default value is used.\n"
            "This parameter should be used in conjunction with --method and "
            "is mutually exclusive with --conf."
        ),
        default=10,
        type=int,
        metavar="INT",
    )
    conf_vs_separate.add_argument(
        "--conf",
        help=(
            "Denoising configuration.\n"
            "Format should be "
            "<name>_<patch-size>_<patch-overlap>_<recombination>_<mask_threshold>.\n"
            "See Documentation of 'DenoiseParameter.from_str' for full specification.\n"
            f"Available denoising methods:\n  {DENOISER_NAMES}.\n"
            "This parameter is mutually exclusive with --method."
        ),
        default=None,
    )

    denoising_group.add_argument(
        "--extra",
        metavar="key=value",
        default=None,
        nargs="+",
        help="extra key=value arguments for denoising methods.",
        action=ToDict,
    )

    data_group = parser.add_argument_group("Additional input data")
    data_group.add_argument(
        "--mask",
        metavar="FILE|auto",
        default=None,
        help=("mask file, if auto, it would be determined from the average image."),
    )
    data_group.add_argument(
        "--noise-map",
        metavar="FILE",
        default=None,
        type=IsFile,
        help="noise map estimation file",
    )
    data_group.add_argument(
        "--input-phase",
        metavar="FILE",
        default=None,
        type=IsFile,
        help=(
            "Phase of the input data. This MUST be in radians. "
            "No rescaling will be applied."
        ),
    )
    data_group.add_argument(
        "--noise-map-phase",
        metavar="FILE",
        default=None,
        type=IsFile,
        help=(
            "Phase component of the noise map estimation file. "
            "This MUST be in radians. No rescaling will be applied."
        ),
    )

    misc_group = parser.add_argument_group("Miscellaneous options")
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
    return parser


def parse_args():
    """Parse input arguments."""
    parser = _get_parser()
    args = parser.parse_args()

    # default value for output.
    if args.output_file is None:
        args.output_file = args.input_file.with_stem("D" + args.input_file.stem)

    if not args.extra:
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
    else:
        input_data, affine = load_as_array(args.input_file)

    kwargs = args.extra

    if args.nan_to_num is not None:
        input_data = np.nan_to_num(input_data, nan=args.nan_to_num)

    logging.info(f"Input data shape: {input_data.shape}")
    n_nans = np.isnan(input_data).sum()
    if n_nans > 0:
        logging.warning(
            f"{n_nans}/{input_data.size} voxels are NaN. "
            "You might want to use --nan-to-num=<value>",
            stacklevel=0,
        )

    if args.mask == "auto":
        mask = compute_mask(input_data)
        affine_mask = None
    else:
        mask, affine_mask = load_as_array(args.mask)

    if args.noise_map is not None and args.noise_map_phase is not None:
        noise_map, affine_noise_map = load_complex_nifti(
            args.noise_map,
            args.noise_map_phase,
        )
    elif args.noise_map is not None:
        noise_map, affine_noise_map = load_as_array(args.noise_map)
    elif args.noise_map_phase is not None:
        raise ValueError(
            "The phase component of the noise map has been provided, "
            "but not the magnitude."
        )
    else:
        noise_map = None
        affine_noise_map = None

    if affine is not None:
        if (affine_mask is not None) and not np.allclose(affine, affine_mask):
            logging.warning(
                "Affine matrix of input and mask does not match", stacklevel=2
            )

        if (affine_noise_map is not None) and not np.allclose(affine, affine_noise_map):
            logging.warning(
                "Affine matrix of input and noise map does not match", stacklevel=2
            )

    # Parse configuration string instead of defining each parameter separately
    if args.conf is not None:
        d_par = DenoiseParameters.from_str(args.conf)
        args.method = d_par.method
        args.patch_shape = d_par.patch_shape
        args.patch_overlap = d_par.patch_overlap
        args.recombination = d_par.recombination
        args.mask_threshold = d_par.mask_threshold

    logging.info(f"Current Setup {args}")
    denoise_func = DENOISER_MAP[args.method]

    if args.method in [
        "nordic",
        "hybrid-pca",
        "adaptive-qut",
        "optimal-fro-noise",
    ]:
        kwargs["noise_std"] = noise_map
        if noise_map is None:
            raise RuntimeError("A noise map must be specified for this method.")

    denoised_data, _, noise_std_map, _ = denoise_func(
        input_data,
        patch_shape=args.patch_shape,
        patch_overlap=args.patch_overlap,
        mask=mask,
        mask_threshold=args.mask_threshold,
        recombination=args.recombination,
        **kwargs,
    )

    save_array(denoised_data, affine, args.output_file)
    save_array(noise_std_map, affine, args.output_noise_map)


if __name__ == "__main__":
    main()
