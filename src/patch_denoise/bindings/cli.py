#!/usr/bin/env python3
"""Cli interface."""

import argparse
import json
import logging
import re
from functools import partial
from pathlib import Path

import numpy as np
from nilearn.image import load_img
from nilearn.interfaces.bids import get_bids_files, parse_bids_filename
from nilearn.interfaces.bids.utils import bids_entities, create_bids_filename
from nilearn.maskers import NiftiMasker

from patch_denoise import __version__
from patch_denoise.bindings.utils import (
    DENOISER_MAP,
    DenoiseParameters,
    load_as_array,
    load_complex_nifti,
    save_array,
)

GPU_AVAILABLE = True
try:
    from patch_denoise.gpu import main_gpu
except ImportError:
    GPU_AVAILABLE = False


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
    return _extend_parser(parser)


def _bids_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help="The directory with the input dataset formatted according to the "
        "BIDS standard.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files should be stored.",
    )
    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. Only participant"
        "level is available.",
        choices=["participant"],
    )
    parser.add_argument(
        "--participant-label",
        "--participant_label",
        help="The label(s) of the participant(s) to analyze. The "
        "label corresponds to sub-<participant-label> from the BIDS spec (so "
        "it does not include 'sub-'). If this parameter is not provided all "
        "subjects will be analyzed. Multiple participants can be specified "
        "with a space separated list.",
        nargs="+",
    )
    parser.add_argument(
        "--session-label",
        "--session_label",
        help="The label(s) of the session(s) to analyze. The "
        "label corresponds to ses-<participant-label> from the BIDS spec (so "
        "it does not include 'ses-'). If this parameter is not provided all "
        "sessions will be analyzed. Multiple sessions can be specified "
        "with a space separated list.",
        nargs="+",
    )
    parser.add_argument(
        "--task-label",
        "--task_label",
        help="The label(s) of the task(s) to analyze. The "
        "label corresponds to task-<participant-label> from the BIDS spec (so "
        "it does not include 'task-'). If this parameter is not provided all "
        "tasks will be analyzed. Multiple tasks can be specified "
        "with a space separated list.",
        nargs="+",
    )
    parser.add_argument(
        "--bids-filter-file",
        type=Path,
        help="A JSON file describing custom BIDS input filters using PyBIDS. "
        "We use the same format as described in fMRIPrep documentation: "
        "https://fmriprep.org/en/latest/faq.html#"
        "how-do-i-select-only-certain-files-to-be-input-to-fmriprep "
        "\nHowever, the query filed should always be 'bold'",
    )
    return _extend_parser(parser, bids_app=True)


def _extend_parser(parser, bids_app=False):
    IsFile = partial(_is_file, parser=parser)
    TuplePositiveInt = partial(_tuple_positive_int, is_parser=True)

    parser.add_argument("--version", action="version", version=__version__)

    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU for computation if available."
    )

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

    if not bids_app:
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
    if not bids_app:
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

    masker = NiftiMasker(verbose=args.verbose, mask_strategy="epi")
    if args.mask != "auto":
        NiftiMasker.mask_img = args.mask

    masker.fit(args.input_file)
    mask = masker.mask_img_.get_fdata().astype(bool)
    affine_mask = masker.mask_img_.affine

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

    if args.output_file is not None:
        parent_dir = Path(args.output_file).parent
        if not Path(args.output_file).parent.exists():
            parent_dir.mkdir(exist_ok=True, parents=True)
            logging.info(f"{Path(args.output_file).parent} created")
        if not Path(args.output_file).exists():
            logging.warn(f"{Path(args.output_file).parent} will be overwritten")

    report = masker.generate_report()
    report.save_as_html(Path(args.output_file).with_suffix(".html"))

    mask_filename = args.output_file.with_stem("mask_" + args.output_file.stem)
    masker.mask_img_.to_filename(mask_filename)

    if args.gpu:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "GPU support is not available. Please ensure that the "
                "patch_denoise.gpu module is installed and that you have"
                "a compatible GPU."
            )
        logging.info("Using GPU for computation.")
        denoised_data, _, noise_std_map = main_gpu(
            args, input_data, mask, noise_map, **kwargs
        )

    else:
        if args.method in [
            "nordic",
            "hybrid-pca",
            "adaptive-qut",
            "optimal-fro-noise",
        ]:
            if noise_map is None:
                raise RuntimeError("A noise map must be specified for this method.")
            kwargs["noise_std"] = noise_map
        denoise_func = DENOISER_MAP.get(args.method, None)
        if denoise_func is None:
            raise ValueError(f"Method {args.method} is not supported.")

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


def bids_app():
    """Run CLI for bids app."""
    parser = _bids_parser()
    args = parser.parse_args()

    if not args.extra:
        args.extra = {}
    kwargs = args.extra

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(args.verbose, len(levels) - 1)]  # cap to last level index
    logging.basicConfig(level=level)

    if args.participant_label:
        all_subjects = args.participant_label
    else:
        all_subjects = [
            x.name.strip("sub-")
            for x in Path(args.bids_dir).iterdir()
            if "sub-" in x.name
        ]

    filters = []
    if args.task_label:
        filters.append(("task", args.task_label[0]))
    if args.session_label:
        filters.append(("ses", args.session_label[0]))

    # Parse configuration string instead of defining each parameter separately
    if args.conf is not None:
        d_par = DenoiseParameters.from_str(args.conf)
        args.method = d_par.method
        args.patch_shape = d_par.patch_shape
        args.patch_overlap = d_par.patch_overlap
        args.recombination = d_par.recombination
        args.mask_threshold = d_par.mask_threshold

    logging.info(f"Current Setup {args}")

    noise_map = None

    if args.gpu and not GPU_AVAILABLE:
        raise RuntimeError(
            "GPU support is not available. Please ensure that the "
            "patch_denoise.gpu module is installed and that you have"
            "a compatible GPU."
        )
    else:
        if args.method in [
            "nordic",
            "hybrid-pca",
            "adaptive-qut",
            "optimal-fro-noise",
        ]:
            if noise_map is None:
                raise RuntimeError("A noise map must be specified for this method.")
            kwargs["noise_std"] = noise_map
        denoise_func = DENOISER_MAP.get(args.method, None)
        if denoise_func is None:
            raise ValueError(f"Method {args.method} is not supported.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ds_json = output_dir / "dataset_description.json"
    GeneratedBy = {
        "Name": "nilearn",
        "Version": __version__,
        "Description": ("A dataset denoised with patch-denoise."),
        "CodeURL": "",
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
            args.bids_dir,
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

            masker = NiftiMasker(verbose=args.verbose, mask_strategy="epi")
            masker.fit(f)
            mask = masker.mask_img_.get_fdata().astype(bool)
            affine = masker.mask_img_.affine

            report = masker.generate_report()
            report.save_as_html(output_mask_filename.with_suffix(".html"))

            masker.mask_img_.to_filename(output_mask_filename)

            input_data = load_img(f).get_fdata()

            if args.gpu:
                denoised_data, _, noise_std_map = main_gpu(
                    args,
                    input_data,
                    mask,
                    # noise_map,
                    # **kwargs
                )
            else:
                denoised_data, _, noise_std_map, _ = denoise_func(
                    input_data,
                    patch_shape=args.patch_shape,
                    patch_overlap=args.patch_overlap,
                    mask=mask,
                    mask_threshold=args.mask_threshold,
                    recombination=args.recombination,
                    # **kwargs,
                )

            print(noise_std_map.shape)
            save_array(denoised_data, affine, output_filename)
            save_array(noise_std_map, affine, output_std_filename)


if __name__ == "__main__":
    main()
