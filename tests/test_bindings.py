"""Test for the binding module."""
import os
import numpy as np
import numpy.testing as npt
import pytest

MODOPT_AVAILABLE = True
NIPYPE_AVAILABLE = True
try:
    import modopt
except ImportError as e:
    MODOPT_AVAILABLE = False
try:
    import nipype
    import nibabel as nib
except ImportError as e:
    NIPYPE_AVAILABLE = False


from patch_denoise.bindings.modopt import LLRDenoiserOperator
from patch_denoise.bindings.nipype import PatchDenoise
from patch_denoise.bindings.utils import DenoiseParameters
from patch_denoise.denoise import mp_pca


@pytest.fixture(scope="module")
def denoised_ref(noisy_phantom):
    denoised_func, _, _ = mp_pca(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        threshold_scale=2.3,
        recombination="weighted",
    )
    return denoised_func


@pytest.fixture
def nifti_noisy_phantom(noisy_phantom, tmpdir_factory):
    tempdir = tmpdir_factory.mktemp("test")
    tempdir.chdir()
    nii_img = nib.Nifti1Image(noisy_phantom, affine=np.eye(4))
    nib.nifti1.save(nii_img, "noisy_phantom.nii")
    return os.path.abspath("noisy_phantom.nii")


def test_modopt(noisy_phantom, denoised_ref):
    """test the Modopt Operator."""
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
    print(nifti_noisy_phantom, tempdir)
    exit_status = os.system(
        f"patch-denoise {nifti_noisy_phantom} {outfile} --conf mp-pca_6_5_w --extra threshold_scale=2.3"
    )
    assert exit_status == 0
    npt.assert_allclose(nib.load(outfile).get_fdata(), denoised_ref)


def test_denoise_param():
    """Test the Denoise parameter structure."""
    d = DenoiseParameters("optimal-fro", 11, 10, "weighted", 10)
    d2 = DenoiseParameters.from_str(str(d))
    assert d2 == d


def test_nipype_mag(nifti_noisy_phantom, denoised_ref):
    """Test the Nipye Interfaces."""

    interface = PatchDenoise()
    interface.inputs.in_mag = nifti_noisy_phantom
    interface.inputs.denoise_str = "mp-pca_6_5_w"
    interface.inputs.extra_kwargs = {"threshold_scale": 2.3}

    output_file = interface.run().outputs.denoised_file

    output_data = nib.load(output_file).get_fdata()
    npt.assert_allclose(output_data, denoised_ref, rtol=1e-2)


def test_nipype_cpx(nifti_noisy_phantom):
    """Test the Nipye Interfaces."""

    interface = PatchDenoise()
    interface.inputs.in_real = nifti_noisy_phantom
    interface.inputs.in_imag = nifti_noisy_phantom
    interface.inputs.denoise_str = "mp-pca_6_5_w"
    interface.inputs.extra_kwargs = {"threshold_scale": 2.3}

    output_file = interface.run().outputs.denoised_file
