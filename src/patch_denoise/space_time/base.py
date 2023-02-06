"""Base Structure for patch-based denoising on spatio-temporal dimension."""
import abc

import numpy as np
from tqdm.auto import tqdm

from .._docs import fill_doc

from .utils import get_patch_locs


@fill_doc
class BaseSpaceTimeDenoiser(abc.ABC):
    """
    Base Class for Patch-based denoising methods for dynamical data.

    Parameters
    ----------
    $patch_config
    """

    def __init__(self, patch_shape, patch_overlap, recombination="weighted"):
        self.p_shape = patch_shape
        self.p_ovl = patch_overlap
        self.recombination = recombination

        self.input_denoising_kwargs = dict()

    @fill_doc
    def denoise(self, input_data, mask=None, mask_threshold=50, progbar=None):
        """Denoise the input_data, according to mask.

        Patches are extracted sequentially and process by the implemented
        `_patch_processing` function.
        Only patches which have at least a voxel in the mask ROI are processed.

        Parameters
        ----------
        $input_config
        $mask_config

        Returns
        -------
        $denoise_return
        """
        data_shape = input_data.shape
        output_data = np.zeros_like(input_data)
        # Create Default mask
        if mask is None:
            process_mask = np.full(data_shape[:-1], True)
        else:
            process_mask = np.copy(mask)

        patch_shape, patch_overlap = self.__get_patch_param(data_shape)
        patch_size = np.prod(patch_shape)

        if self.recombination == "center":
            patch_center = (
                *(slice(ps // 2, ps // 2 + 1) for ps in patch_shape),
                slice(None, None, None),
            )
        patchs_weight = np.zeros(data_shape[:-1], np.float32)
        noise_std_estimate = np.zeros(data_shape[:-1], dtype=np.float32)

        # discard useless patches
        patch_locs = get_patch_locs(patch_shape, patch_overlap, data_shape[:-1])
        get_it = np.zeros(len(patch_locs), dtype=bool)

        for i, patch_tl in enumerate(patch_locs):
            patch_slice = tuple(
                slice(tl, tl + ps) for tl, ps in zip(patch_tl, patch_shape)
            )
            if 100 * np.sum(process_mask[patch_slice]) / patch_size > mask_threshold:
                get_it[i] = True

        print("Denoise {:.2f}% patches".format(100 * np.sum(get_it) / len(patch_locs)))
        patch_locs = np.ascontiguousarray(patch_locs[get_it])

        if progbar is None:
            progbar = tqdm(total=len(patch_locs))
        elif progbar is not False:
            progbar.reset(total=len(patch_locs))

        for patch_tl in patch_locs:
            patch_slice = tuple(
                slice(tl, tl + ps) for tl, ps in zip(patch_tl, patch_shape)
            )
            process_mask[patch_slice] = 1
            # building the casoratti matrix
            patch = np.reshape(input_data[patch_slice], (-1, input_data.shape[-1]))

            # Replace all nan by mean value of patch.
            # FIXME this behaviour should be documented
            # And ideally choosen by the user.

            patch[np.isnan(patch)] = np.mean(patch)
            p_denoise, *extras = self._patch_processing(
                patch,
                patch_slice=patch_slice,
                **self.input_denoising_kwargs,
            )

            p_denoise = np.reshape(p_denoise, (*patch_shape, -1))
            if self.recombination == "center":
                patch_center_img = tuple(
                    ptl + ps // 2 for ptl, ps in zip(patch_tl, patch_shape)
                )
                output_data[patch_center_img] = p_denoise[patch_center]
                patchs_weight[patch_center_img] += extras[0]
                noise_std_estimate[patch_center_img] += extras[1]
            else:
                output_data[patch_slice] += p_denoise
                if self.recombination == "weighted":
                    patchs_weight[patch_slice] += extras[0]
                elif self.recombination == "average":
                    patchs_weight[patch_slice] += 1
            if len(extras) > 1:
                noise_std_estimate[patch_slice] += extras[1]
            if progbar:
                progbar.update()
        # Averaging the overlapping pixels.
        output_data /= patchs_weight[..., None]
        noise_std_estimate /= patchs_weight

        output_data[~process_mask] = 0

        return output_data, patchs_weight, noise_std_estimate

    @abc.abstractmethod
    def _patch_processing(self, patch, patch_slice=None, **kwargs):
        """Process a patch.

        Implemented by child classes.
        """

    def __get_patch_param(self, data_shape):
        """Return tuple for patch_shape and patch_overlap.

        It works from whatever the  input format was (int or list).
        This method also ensure that the patch will provide tall and skinny matrices.
        """
        pp = [None, None]
        for i, attr in enumerate(["p_shape", "p_ovl"]):
            p = getattr(self, attr)
            if isinstance(p, list):
                p = tuple(p)
            elif isinstance(p, (int, np.integer)):
                p = (p,) * (len(data_shape) - 1)
            pp[i] = p
        if np.prod(pp[0]) < data_shape[-1]:
            raise ValueError(
                f"the number of voxel in patch ({np.prod(pp[0])}) is smaller than the"
                f" last dimension ({data_shape[-1]}), this makes an ill-conditioned"
                "matrix for SVD."
            )
        return tuple(pp)
