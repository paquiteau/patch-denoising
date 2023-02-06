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
except ImportError as e:
    NIPYPE_AVAILABLE = False


from patch_denoise.bindings.modopt import LLRDenoiserOperator
from patch_denoise.denoise import mp_pca


def test_modopt(noisy_phantom):
    """test the Modopt Operator."""
    operator = LLRDenoiserOperator(
        "mp-pca",
        patch_shape=6,
        patch_overlap=5,
        threshold_scale=2.3,
        recombination="weighted",
    )

    denoised_modopt = operator.op(noisy_phantom)
    denoised_func, _, _ = mp_pca(
        noisy_phantom,
        patch_shape=6,
        patch_overlap=5,
        threshold_scale=2.3,
        recombination="weighted",
    )

    npt.assert_allclose(denoised_modopt, denoised_func)
