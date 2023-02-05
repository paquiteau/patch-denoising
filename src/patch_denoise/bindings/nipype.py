"""Nipype Bindings, Provide bindings To apply patch based denoising."""

import os

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import split_filename

from .utils import DENOISER_MAP, DenoiseParameters
from ..space_time.utils import estimate_noise


class PatchDenoiseInputSpec(BaseInterfaceInputSpec):
    """InputSpec for Patch denoising Interface."""

    in_mag = File(
        exists=True,
        xor=["in_real", "in_imag"],
        desc="magnitude input file to denoise.",
    )
    in_real = File(
        exists=True,
        xor=["in_mag"],
        require=["in_imag"],
        desc="Real-part of input file to denoise.",
    )
    in_imag = File(
        exists=True,
        xor=["in_mag"],
        require=["in_real"],
        desc="imaginary part of input file to denoise.",
    )

    mask = File(exists=True)
    noise_std_map = File(desc="noise_std_map")
    denoise_str = traits.Str(desc="string describing the denoiser configuration")
    method = traits.Enum(
        *DENOISER_MAP.keys(),
        xor=["denoise_str"],
        require=["patch_shape", "patch_overlap"],
    )
    patch_shape = traits.Union(
        traits.Int(),
        traits.List(traits.Int(), minlen=3, maxlen=3),
        xor=["denoise_str"],
        require=["denoise_method", "patch_overlap"],
    )
    patch_overlap = traits.Union(
        traits.Int(),
        traits.List(traits.Int(), minlen=3, maxlen=3),
        xor=["denoise_str"],
        require=["patch_shape", "denoise_method"],
    )
    mask_threshold = traits.Int(10)
    recombination = traits.Enum("weighted", "mean")
    extra_kwargs = traits.Dict()


class PatchDenoiseOutputSpec(TraitedSpec):
    """OutputSpec for Denoising Interface."""

    denoised_file = File(desc="denoised file")
    noise_std_map = File(desc="a map of the noise variance.")
    pass


class PatchDenoise(SimpleInterface):
    """Patch based denoising interface."""

    input_spec = PatchDenoiseInputSpec
    output_spec = PatchDenoiseOutputSpec

    _denoise_attrs = [
        "method",
        "patch_shape",
        "patch_overlap",
        "mask_threshold",
        "recombination",
    ]

    def _run_interface(self, runtime):
        # INPUT
        if isdefined(self.inputs.denoise_str):
            d_par = DenoiseParameters.from_str(self.inputs.denoise_str)
        else:
            d_par = DenoiseParameters()
            for attr in PatchDenoise._denoise_attrs:
                setattr(d_par, attr, getattr(self.inputs, attr))

        if isdefined(self.inputs.in_mag):
            data_mag_nii = nib.load(self.inputs.in_mag)
            data = data_mag_nii.get_fdata(dtype=np.float32)
            basename = self.inputs.in_mag
            affine = data_mag_nii.affine
        else:
            data_real_nii = nib.load(self.inputs.in_real)
            affine = data_real_nii.affine
            data_real = data_real_nii.get_fdata(dtype=np.float32)
            data_imag = nib.load(self.inputs.in_imag).get_fdata(dtype=np.float32)
            data = 1j * data_imag
            data += data_real
            basename = self.inputs.in_real

        if isdefined(self.inputs.mask) and self.inputs.mask:
            mask = np.abs(nib.load(self.inputs.mask).get_fdata()) > 0
        else:
            mask = None

        try:
            denoise_func = DENOISER_MAP[d_par.method]
        except KeyError:
            raise ValueError(
                f"unknown denoising denoise_method '{self.inputs.denoise_method}', "
                f"available are {list(DENOISER_MAP.keys())}"
            ) from None

        if isdefined(self.inputs.extra_kwargs) and self.inputs.extra_kwargs:
            extra_kwargs = self.inputs.extra_kwargs
        else:
            extra_kwargs = dict()
        if d_par.method in [
            "nordic",
            "hybrid-pca",
            "adaptive-qut",
            "optimal-fro-noise",
        ]:
            extra_kwargs["noise_std"] = nib.load(self.inputs.noise_std_map).get_fdata()

        if denoise_func is not None:
            # CORE CALL
            denoised_data, _, noise_std_map = denoise_func(
                data,
                patch_shape=d_par.patch_shape,
                patch_overlap=d_par.patch_overlap,
                mask=mask,
                mask_threshold=d_par.mask_threshold,
                recombination=d_par.recombination,
                **extra_kwargs,
            )
        else:
            denoised_data = data
            noise_std_map = np.std(data, axis=-1, dtype=np.float32)
        # OUTPUT
        if np.any(np.iscomplex(denoised_data)):
            denoised_data = np.abs(denoised_data, dtype=np.float32)

        _, base, _ = split_filename(basename)
        base = base.replace("_mag", "")
        base = base.replace("_real", "")
        denoise_filename = f"{base}_d_{d_par.method}.nii"
        noise_map_filename = f"{base}_noise_map.nii"

        denoised_data_img = nib.Nifti1Image(denoised_data, affine=affine)
        denoised_data_img.to_filename(denoise_filename)

        noise_map_img = nib.Nifti1Image(noise_std_map, affine=affine)
        noise_map_img.to_filename(noise_map_filename)

        self._results["denoised_file"] = os.path.abspath(denoise_filename)
        self._results["noise_std_map"] = os.path.abspath(noise_map_filename)

        return runtime


class NoiseStdMapInputSpec(BaseInterfaceInputSpec):
    """InputSpec for Noise Map Estimation."""

    noise_map_file = File(
        exists=True,
        mandatory=True,
        desc="A 0-Volt volume acquisition",
    )
    fft_scale = traits.Int(default=100, desc="scaling parameter of the reconstruction.")
    block_size = traits.Int(default=3, desc="size of spatial block to compute the std.")


class NoiseStdMapOutputSpec(TraitedSpec):
    """OutputSpec for Noise Map Estimation."""

    noise_std_map = File(desc="Spatial variation of noise variance")


class NoiseStdMap(SimpleInterface):
    """Noise std estimation."""

    input_spec = NoiseStdMapInputSpec
    output_spec = NoiseStdMapOutputSpec

    def _run_interface(self, runtime):
        noise_map = nib.load(self.inputs.noise_map_file)
        noise_std_map = estimate_noise(
            noise_map.get_fdata() / self.inputs.fft_scale, self.inputs.block_size
        )
        noise_std_map_img = nib.Nifti1Image(noise_std_map, affine=noise_map.affine)

        filename = os.path.abspath(
            os.path.basename(self.inputs.noise_map_file).split(".")[0] + "_std.nii"
        )
        noise_std_map_img.to_filename(filename)
        self._results["noise_std_map"] = filename

        return runtime
