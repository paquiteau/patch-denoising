#!/usr/bin/env python3
"""Cli interface."""


import argparse
import warnings

import numpy as np

from .utils import DENOISER_MAP, DenoiseParameters

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
            "Format should be <name>_<patch-size>_<patch-overlap>_<recombination>.\n"
            "See Documentation of ``DenoiseParameter.from_str`` for full specification."
            "Default:\n   optimal-fro_11_5_w\n"
            "Available denoising methods:\n  " + DENOISER_NAMES,
        ),
        default="optimal-fro_11_5_w",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="mask file",
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

    args = parser.parse_args()

    # default value for output.
    if args.output is None:
        input_path = args.input.split("/")
        args.output = "/".join(input_path)[:-1] + "/d" + input_path[-1]


def load_as_array(input):
    if input is None:
        return None
    if input.endswith(".npy"):
        return np.load(input)
    elif input.endswith(".nii") or input.endswith(".nii.gz"):
        return nib.load(input).get_fdata()
    else:
        raise ValueError("Unsupported file format. use numpy or nifti formats.")


def save_array(data, affine, filename):
    if filename is None:
        return None

    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        if affine is None:
            affine = np.eye(len(data.shape))
        nii_img = nib.Nifti1Image(data, affine)
        nii_img.save(filename)
    elif filename.endswith(".npy"):
        np.save(filename, data)

    return filename


def main():
    args = parse_args()
    print(args)

    input_data, affine = load_as_array(args.input)
    mask, affine_mask = load_as_array(args.mask)
    noise_map, affine_noise_map = load_as_array(args.noise_map)

    if affine is not None:
        if affine_mask is not None and np.allclose(affine, affine_mask):
            warnings.warn("Affine matrix of input and mask does not match")
        if affine_noise_map is not None and np.allclose(affine, affine_noise_map):
            warnings.warn("Affine matrix of input and noise map does not match")
    d_par = DenoiseParameters.from_str(args.conf)

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
    denoised_data, _, noise_std_map = denoise_func(
        input_data,
        patch_shape=d_par.patch_shape,
        patch_overlap=d_par.patch_overlap,
        mask=mask,
        mask_threshold=d_par.mask_threshold,
        recombination=d_par.recombination,
        **extra_kwargs,
    )

    save_array(denoised_data, affine, args.output)
    save_array(noise_std_map, affine, args.output_noise_map)


if __name__ == "__main__":
    main()