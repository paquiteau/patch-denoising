"""Binding for ModOpt Package."""

import numpy as np
from modopt.opt.proximity import ProximityParent

from .utils import DENOISER_MAP


class LLRDenoiserOperator(ProximityParent):
    """Proximal Operator drop-in replacement using local low rank denoising.

    Parameters
    ----------
    denoiser: str
        name of the denoising method.
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
        will be processed.
    mask_threshold: int
        percentage of the path that should be in the mask in order to be processed.
        If mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
        the voxels of the patch needs to be in the mask
    """

    def __init__(
        self,
        denoiser,
        patch_shape,
        patch_overlap,
        recombination="weighted",
        mask=None,
        mask_threshold=-1,
        progbar=None,
        time_dimension=-1,
        **kwargs,
    ):
        self._denoiser = DENOISER_MAP[denoiser]
        self._params = dict(
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
            mask=mask,
            mask_threshold=mask_threshold,
            progbar=progbar,
            **kwargs,
        )
        self.op = self._op_method
        self.cost = lambda *args, **kw: np.NaN
        self.time_dimension = time_dimension

    def _op_method(self, data, **kwargs):
        run_kwargs = self._params.copy()
        run_kwargs.update(kwargs)
        return np.moveaxis(
            self._denoiser(
                np.moveaxis(data, self.time_dimension, -1),
                **run_kwargs,
            )[0],
            -1,
            self.time_dimension,
        )
