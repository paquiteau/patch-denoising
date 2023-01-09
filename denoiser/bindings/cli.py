#!/usr/bin/env python3
"""Cli interface."""


import argparse

import numpy as np

from .utils import DENOISER_MAP

NIBABEL_AVAILABLE = True
try:
    import nibabel as nib
except ImportError:
    NIBABEL_AVAILABLE = False


DENOISER_NAMES = ", ".join(d for d in DENOISER_MAP if d)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_file", help="Input (noisy) file.")
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help=("Output (denoised) file.\n" "default is d<input_file_name>"),
    )

    parser.add_argument(
        "--conf",
        help=(
            "denoising configuration.\n"
            "Format should be <method-name>_<patch-size>_<patch-overlap>_<recombination-method>.\n"
            "Default:\n   optimal-fro_11_5_w\n"
            "Available denoising methods:\n  " + DENOISER_NAMES
        ),
    )
    parser.add_argument("--mask")

    args = parser.parse_args()

    # default value for output.
    if args.output is None:
        input_path = args.input.split("/")
        args.output = "/".join(input_path)[:-1] + "/d" + input_path[-1]


def load_as_array(input):
    if input.endswith(".npy"):
        return np.load(input)
    elif input.endswith(".nii") or input.endswith(".nii.gz"):
        return nib.load(input).get_fdata()
    else:
        raise ValueError("Unsupported file format. use numpy or nifti formats.")


def main():
    args = parse_args()
    print(args)

    input_data = load_as_array(args.input)


if __name__ == "__main__":
    main()
