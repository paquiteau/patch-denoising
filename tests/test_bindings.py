"""Test for the binding module."""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from patch_denoise.bindings.cli import main

MODOPT_AVAILABLE = True
NIPYPE_AVAILABLE = True
try:
    import modopt
except ImportError as e:
    MODOPT_AVAILABLE = False
try:
    import nibabel as nib
    import nipype
except ImportError as e:
    NIPYPE_AVAILABLE = False


from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.bindings.modopt import LLRDenoiserOperator
from patch_denoise.bindings.nipype import PatchDenoise
from patch_denoise.bindings.utils import DenoiseParameters
from patch_denoise.denoise import mp_pca


@pytest.fixture(scope="module")
def denoised_ref(noisy_phantom):
    return mp_pca(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        threshold_scale=2.3,
        recombination="weighted",
    )[0]


@pytest.fixture
def nifti_noisy_phantom(noisy_phantom, tmpdir_factory):
    tempdir = tmpdir_factory.mktemp("test")
    tempdir.chdir()
    nii_img = nib.Nifti1Image(noisy_phantom, affine=np.eye(4))
    nib.nifti1.save(nii_img, "noisy_phantom.nii")
    return os.path.abspath("noisy_phantom.nii")


def test_modopt(noisy_phantom, denoised_ref):
    """Test the Modopt Operator."""
    operator = LLRDenoiserOperator(
        "mp-pca",
        patch_shape=6,
        patch_overlap=5,
        threshold_scale=2.3,
        recombination="weighted",
    )

    denoised_modopt = operator.op(noisy_phantom)

    npt.assert_allclose(denoised_modopt, denoised_ref)


def test_entrypoint():
    """Test entrypoint of patch-denoise function."""
    exit_status = os.system("patch-denoise --help")
    assert exit_status == 0


def test_cli(nifti_noisy_phantom, tmpdir_factory, denoised_ref):
    tempdir = tmpdir_factory.mktemp("test")
    tempdir.chdir()
    outfile = "out.nii"
    exit_status = os.system(
        f"patch-denoise {nifti_noisy_phantom} {outfile} --conf mp-pca_6_5_weighted --extra threshold_scale=2.3"
    )
    assert exit_status == 0
    npt.assert_allclose(
        nib.load(outfile).get_fdata(dtype=np.float32),
        denoised_ref,
        rtol=1e-6,
        atol=1e-2,
    )


def test_denoise_param():
    """Test the Denoise parameter structure."""
    d = DenoiseParameters("optimal-fro", 11, 10, "weighted", 10)
    d2 = DenoiseParameters.from_str(str(d))
    assert d2 == d


def test_nipype_mag(nifti_noisy_phantom, denoised_ref):
    """Test the Nipye Interfaces."""
    interface = PatchDenoise()
    interface.inputs.in_mag = nifti_noisy_phantom
    interface.inputs.denoise_str = "mp-pca_6_5_weighted"
    interface.inputs.extra_kwargs = {"threshold_scale": 2.3}

    output_file = interface.run().outputs.denoised_file

    output_data = nib.load(output_file).get_fdata(dtype=np.float32)
    npt.assert_allclose(output_data, denoised_ref, rtol=1e-2)


def test_nipype_cpx(nifti_noisy_phantom):
    """Test the Nipye Interfaces."""
    interface = PatchDenoise()
    interface.inputs.in_real = nifti_noisy_phantom
    interface.inputs.in_imag = nifti_noisy_phantom
    interface.inputs.denoise_str = "mp-pca_6_5_weighted"
    interface.inputs.extra_kwargs = {"threshold_scale": 2.3}

    output_file = interface.run().outputs.denoised_file


def test_denoise_paramter_pretty_par():
    pretty_par = DenoiseParameters("optimal-fro", 11, 10, "weighted", 10).pretty_par

    assert pretty_par == "11_10w"


def test_denoise_parameter_pretty():
    """Test the pretty_name."""
    pretty_string = "optimal-fro_11_10_weighted_10"
    pretty_name = DenoiseParameters.from_str(pretty_string).pretty_name

    assert pretty_name == pretty_string


@pytest.fixture
def data() -> Path:
    """Path to real data."""
    return Path(__file__).parent / "data"


@pytest.fixture
def ds001168(data) -> Path:
    """Real BIDS dataset used for end to end testing.

    run 'tox run -e data' to datalad install and get the relevant files
    """
    return data / "ds001168"


@pytest.mark.e2e
def test_e2e(data, ds001168):
    input_file = f"{ds001168}/sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz"
    output_file = f"{data}/derivatives/sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_desc-denoised_bold.nii.gz"
    exit_status = os.system(
        f"patch-denoise {input_file} {output_file} --conf mp-pca_10_3_weighted"
    )

    assert exit_status == 0

    if GPU_AVAILABLE:
        import torch

        try:
            torch._C._cuda_init()
            output_file = f"{data}/derivatives/sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_desc-denoised+gpu_bold.nii.gz"
            exit_status = os.system(
                f"patch-denoise {input_file} {output_file} --conf mp-pca_10_3_weighted --gpu"
            )

            assert exit_status == 0
        except RuntimeError:
            print("skipping GPU test")
    else:
        print("skipping GPU test")
