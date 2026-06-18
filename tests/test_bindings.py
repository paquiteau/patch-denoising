"""Test for the binding module."""

import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from patch_denoise.bindings.cli import main

NIPYPE_AVAILABLE = True
try:
    import nipype
except ImportError as e:
    NIPYPE_AVAILABLE = False


from patch_denoise.bindings.cli import GPU_AVAILABLE
from patch_denoise.bindings.nipype import PatchDenoise
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


def test_entrypoint():
    """Test entrypoint of patch-denoise function."""
    exit_status = subprocess.call("patch-denoise --help", shell=True)
    assert exit_status == 0


def test_cli(noisy_phantom, nifti_noisy_phantom, tmpdir_factory, denoised_ref):
    tempdir = tmpdir_factory.mktemp("test")
    tempdir.chdir()
    outfile = "out.nii"

    # we need an explicit mask
    # otherwise the implicit mask will exclude
    # all voxels from the noisy_phantom
    mask = np.ones(noisy_phantom.shape)
    mask_img = nib.Nifti1Image(mask, affine=np.eye(4))
    nib.nifti1.save(mask_img, "mask.nii")

    exit_status = subprocess.call(
        f"patch-denoise {nifti_noisy_phantom} {outfile} --mask mask.nii "
        "-m mp-pca -ps 6 -po 5 -r weighted --extra threshold_scale=2.3",
        shell=True,
    )
    assert exit_status == 0
    npt.assert_allclose(
        nib.Nifti1Image.from_filename(outfile).get_fdata(dtype=np.float32),
        denoised_ref,
        rtol=1e-6,
        atol=1e-2,
    )


def test_nipype_mag(nifti_noisy_phantom, denoised_ref):
    """Test the Nipye Interfaces."""
    interface = PatchDenoise()
    interface.inputs.in_mag = nifti_noisy_phantom
    interface.inputs.method = "mp-pca"
    interface.inputs.patch_shape = 6
    interface.inputs.patch_overlap = 5
    interface.inputs.recombination = "weighted"
    interface.inputs.extra_kwargs = {"threshold_scale": 2.3}

    output_file = interface.run().outputs.denoised_file

    output_data = nib.Nifti1Image.from_filename(output_file).get_fdata(dtype=np.float32)
    npt.assert_allclose(output_data, denoised_ref, rtol=1e-2)


def test_nipype_cpx(nifti_noisy_phantom):
    """Test the Nipye Interfaces."""
    interface = PatchDenoise()
    interface.inputs.in_real = nifti_noisy_phantom
    interface.inputs.in_imag = nifti_noisy_phantom
    interface.inputs.method = "mp-pca"
    interface.inputs.patch_shape = 6
    interface.inputs.patch_overlap = 5
    interface.inputs.recombination = "weighted"
    interface.inputs.extra_kwargs = {"threshold_scale": 2.3}

    output_file = interface.run().outputs.denoised_file


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
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available"),
        ),
    ],
)
def test_e2e(data, ds001168, device):
    input_file = f"{ds001168}/sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz"
    output_file = f"{data}/derivatives/sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_desc-denoised_bold.nii.gz"
    exit_status = subprocess.call(
        f"patch-denoise {input_file} {output_file} -m mp-pca -ps 10 -po 3 -r weighted --{device}",
        shell=True,
    )

    assert exit_status == 0
