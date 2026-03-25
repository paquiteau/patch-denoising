"""Base Structure for patch-based denoising on spatio-temporal dimension."""

import abc
import logging

import numpy as np
from tqdm.auto import tqdm

from .._docs import fill_doc


class PatchedArray:
    """A container for accessing custom view of array easily.

    Parameters
    ----------
    array: np.ndarray
    patch_shape: tuple
    patch_overlap: tuple

    """

    def __init__(
        self,
        array,
        patch_shape,
        patch_overlap,
        dtype=None,
        padding_mode="edge",
        **kwargs,
    ):
        if isinstance(array, tuple):
            array = np.zeros(array, dtype=dtype)
        self._arr = array

        self._ps = np.asarray(patch_shape)
        self._po = np.asarray(patch_overlap)
        self._po = patch_overlap

        dimensions = self._arr.ndim
        step = self._ps - self._po
        if np.any(step < 0):
            raise ValueError("overlap should be smaller than patch on every dimension.")

        if self._ps.size != dimensions or step.size != dimensions:
            raise ValueError(
                "self._ps and step must have the same number of dimensions as the "
                "input self._array."
            )

        # Ensure patch size is not larger than self._array size along each axis
        self._ps = np.minimum(self._ps, self._arr.shape)

        # Calculate the shape and strides of the sliding view
        grid_shape = tuple(
            (
                ((self._arr.shape[i] - self._ps[i]) // step[i] + 1)
                if self._ps[i] < self._arr.shape[i]
                else 1
            )
            for i in range(dimensions)
        )
        shape = grid_shape + tuple(self._ps)
        strides = (
            tuple(
                (
                    self._arr.strides[i] * step[i]
                    if self._ps[i] < self._arr.shape[i]
                    else 0
                )
                for i in range(dimensions)
            )
            + self._arr.strides
        )

        # Create the sliding view
        self.sliding_view = np.lib.stride_tricks.as_strided(
            self._arr, shape=shape, strides=strides
        )

        self._grid_shape = grid_shape

    @property
    def n_patches(self):
        """Get number of patches."""
        return np.prod(self._grid_shape)

    def get_patch(self, idx):
        """Get patch at linear index ``idx``."""
        return self.sliding_view[np.unravel_index(idx, self._grid_shape)]

    def set_patch(self, idx, value):
        """Set patch at linear index ``idx`` with value."""
        self.sliding_view[np.unravel_index(idx, self._grid_shape)]

    def add2patch(self, idx, value):
        """Add to patch, in place."""
        patch = self.get_patch(idx)
        # self.set_patch(idx, patch + value)
        patch += value

    # def sync(self):
    #     """Apply the padded value to the array back."""
    #     np.copyto(
    #         self._array,
    #         self._padded_array[
    #             tuple(
    #                 np.s_[: (s + 1 - ps) if (s - ps) else s]
    #                 for ps, s in zip(self._ps, self._padded_array.shape)
    #             )
    #         ],
    #     )

    # def get(self):
    #     """Return the regular array, after applying the padded values."""
    #     self.sync()
    #     return self._array

    def __getattr__(self, name):
        """Get attribute of underlying array."""
        return getattr(self._arr, name)


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

        if recombination not in ["weighted", "average", "center"]:
            raise ValueError(
                "recombination must be one of 'weighted', 'average', 'center'"
            )

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
        p_s, p_o = self._get_patch_param(data_shape)

        input_data = PatchedArray(input_data, p_s, p_o)
        output_data = PatchedArray(data_shape, p_s, p_o, dtype=input_data.dtype)
        patch_weights = PatchedArray(data_shape, p_s, p_o, dtype=np.float32)
        rank_map = PatchedArray(data_shape, p_s, p_o, dtype=np.int32)
        noise_std_estimate = PatchedArray(data_shape, p_s, p_o, dtype=np.float32)
        # Create Default mask
        if mask is None:
            process_mask = np.full(data_shape, True)
        elif mask.shape == input_data.shape[:-1]:
            process_mask = np.broadcast_to(mask[..., None], input_data.shape)

        process_mask = PatchedArray(
            process_mask, p_s, p_o, padding_mode="constant", constant_values=0
        )

        center_pos = tuple(p // 2 for p in p_s)
        patch_space_size = np.prod(p_s[:-1])
        # select only queue index where process_mask is valid.
        get_it = np.zeros(input_data.n_patches, dtype=bool)

        for i in range(len(get_it)):
            pm = process_mask.get_patch(i)
            if 100 * np.sum(pm) / pm.size > mask_threshold:
                get_it[i] = True

        select_patches = np.nonzero(get_it)[0]
        del get_it

        if progbar is None:
            progbar = tqdm(total=len(select_patches))
        elif progbar is not False:
            progbar.reset(total=len(select_patches))

        for i in select_patches:
            input_patch_casorati = input_data.get_patch(i).reshape(patch_space_size, -1)
            p_denoise, maxidx, noise_var = self._patch_processing(
                input_patch_casorati,
                patch_idx=i,
                **self.input_denoising_kwargs,
            )

            p_denoise = np.reshape(p_denoise, p_s)
            if self.recombination == "center":
                output_data.get_patch(i)[center_pos] = p_denoise[center_pos]
            elif self.recombination == "weighted":
                theta = 1 / (2 + maxidx)
                output_data.add2patch(i, p_denoise * theta)
                patch_weights.add2patch(i, theta)
            elif self.recombination == "average":
                output_data.add2patch(i, p_denoise)
                patch_weights.add2patch(i, 1)
            else:
                raise ValueError(
                    "recombination must be one of 'weighted', 'average', 'center'"
                )
            if progbar:
                progbar.update()
        # Averaging the overlapping pixels.
        # this is only required for averaging recombinations.

        output_data = output_data._arr
        patch_weights = patch_weights._arr

        if self.recombination in ["average", "weighted"]:
            output_data /= patch_weights

        output_data[~process_mask._arr] = 0

        return output_data, patch_weights, noise_std_estimate, rank_map

        # if self.recombination == "center":
        #     patch_center = (
        #         *(slice(ps // 2, ps // 2 + 1) for ps in patch_shape),
        #         slice(None, None, None),
        #     )
        # patchs_weight = np.zeros(data_shape[:-1], np.float32)
        # noise_std_estimate = np.zeros(data_shape[:-1], dtype=np.float32)

        # # discard useless patches
        # patch_locs = get_patch_locs(patch_shape, patch_overlap, data_shape)
        # get_it = np.zeros(len(patch_locs), dtype=bool)

        # for i, patch_tl in enumerate(patch_locs):
        #     patch_slice = tuple(
        #         slice(tl, tl + ps) for tl, ps in zip(patch_tl, patch_shape)
        #     )
        #     if 100 * np.sum(process_mask[patch_slice]) / patch_size > mask_threshold:
        #         get_it[i] = True

        # logging.info(f"Denoise {100 * np.sum(get_it) / len(patch_locs):.2f}% patches")
        # patch_locs = np.ascontiguousarray(patch_locs[get_it])

        # if progbar is None:
        #     progbar = tqdm(total=len(patch_locs))
        # elif progbar is not False:
        #     progbar.reset(total=len(patch_locs))

        # for patch_tl in patch_locs:
        #     patch_slice = tuple(
        #         slice(tl, tl + ps) for tl, ps in zip(patch_tl, patch_shape)
        #     )
        #     process_mask[patch_slice] = 1
        #     # building the casoratti matrix
        #     patch = np.reshape(input_data[patch_slice], (-1, input_data.shape[-1]))

        #     # Replace all nan by mean value of patch.
        #     # FIXME this behaviour should be documented
        #     # And ideally chosen by the user.

        #     patch[np.isnan(patch)] = np.mean(patch)
        #     p_denoise, maxidx, noise_var = self._patch_processing(
        #         patch,
        #         patch_slice=patch_slice,
        #         **self.input_denoising_kwargs,
        #     )

        #     p_denoise = np.reshape(p_denoise, (*patch_shape, -1))
        #     patch_center_img = tuple(
        #         ptl + ps // 2 for ptl, ps in zip(patch_tl, patch_shape)
        #     )
        #     if self.recombination == "center":
        #         output_data[patch_center_img] = p_denoise[patch_center]
        #         noise_std_estimate[patch_center_img] += noise_var
        #     elif self.recombination == "weighted":
        #         theta = 1 / (2 + maxidx)
        #         output_data[patch_slice] += p_denoise * theta
        #         patchs_weight[patch_slice] += theta
        #     elif self.recombination == "average":
        #         output_data[patch_slice] += p_denoise
        #         patchs_weight[patch_slice] += 1
        #     else:
        #         raise ValueError(
        #             "recombination must be one of 'weighted', 'average', 'center'"
        #         )
        #     if not np.isnan(noise_var):
        #         noise_std_estimate[patch_slice] += noise_var
        #     # the top left corner of the patch is used as id for the patch.
        #     rank_map[patch_center_img] = maxidx
        #     if progbar:
        #         progbar.update()
        # # Averaging the overlapping pixels.
        # # this is only required for averaging recombinations.
        # if self.recombination in ["average", "weighted"]:
        #     output_data /= patchs_weight[..., None]
        #     noise_std_estimate /= patchs_weight

        # output_data[~process_mask] = 0

        # return output_data, patchs_weight, noise_std_estimate, rank_map

    @abc.abstractmethod
    def _patch_processing(self, patch, patch_slice=None, **kwargs):
        """Process a patch.

        Implemented by child classes.
        """

    def _get_patch_param(self, data_shape):
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
            if len(p) == len(data_shape) - 1:
                # add the time dimension
                p = (*p, data_shape[-1])
            pp[i] = p

        if np.prod(pp[0][:-1]) < data_shape[-1]:
            logging.warning(
                f"the number of voxel in patch ({np.prod(pp[0])}) is smaller than the"
                f" last dimension ({data_shape[-1]}), this makes an ill-conditioned"
                "matrix for SVD.",
                stacklevel=2,
            )
        return tuple(pp)
