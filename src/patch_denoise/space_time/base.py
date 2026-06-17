"""Base Structure for patch-based denoising on spatio-temporal dimension."""

import abc
import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm.rich import tqdm

from .._docs import fill_doc

log = logging.getLogger(__name__)


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
        array: NDArray,
        patch_shape: tuple[int, ...],
        patch_overlap: tuple[int, ...],
        dtype: DTypeLike | None = None,
        padding_mode: str = "edge",
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
    def n_patches(self) -> int:
        """Get number of patches."""
        return int(np.prod(self._grid_shape))

    def get_patch(self, idx: int) -> NDArray:
        """Get patch at linear index ``idx``."""
        return self.sliding_view[np.unravel_index(idx, self._grid_shape)]

    def set_patch(self, idx: int, value: Any):
        """Set patch at linear index ``idx`` with value."""
        self.sliding_view[np.unravel_index(idx, self._grid_shape)] = value

    def idx2slice(self, idx):
        """Convert linear patch index to slice."""
        grid_idx = np.unravel_index(idx, self._grid_shape)
        return tuple(
            slice(g * (ps - po), g * (ps - po) + ps)
            for g, ps, po in zip(grid_idx, self._ps, self._po)
        )

    @classmethod
    def linear_to_patch_indices(cls, idx, data_shape, patch_shape, patch_overlap):
        """Convert linear patch index to patch indices."""
        grid_shape = tuple(
            (
                (
                    (data_shape[i] - patch_shape[i])
                    // (patch_shape[i] - patch_overlap[i])
                    + 1
                )
                if patch_shape[i] < data_shape[i]
                else 1
            )
            for i in range(len(data_shape))
        )
        grid_idx = np.unravel_index(idx, grid_shape)
        return tuple(
            g * (ps - po) for g, ps, po in zip(grid_idx, patch_shape, patch_overlap)
        )

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

    _patch_processing: Callable[..., tuple[NDArray, int, float]]

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
    def denoise(
        self,
        input_data: NDArray,
        mask: NDArray | None = None,
        mask_threshold: int = 50,
        progbar=None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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
        log.debug(f"Starting denoising process. with {self}")
        data_shape = input_data.shape
        p_s, p_o = self._get_patch_param(data_shape)

        input_data_ = PatchedArray(input_data, p_s, p_o)
        output_data = PatchedArray(data_shape, p_s, p_o, dtype=input_data_.dtype)
        patch_weights = PatchedArray(data_shape, p_s, p_o, dtype=np.float32)
        patch_counts = PatchedArray(data_shape, p_s, p_o, dtype=np.int32)
        rank_map = PatchedArray(data_shape, p_s, p_o, dtype=np.int32)
        noise_std_estimate = PatchedArray(data_shape, p_s, p_o, dtype=np.float32)
        # Create Default mask
        if mask is None:
            process_mask = np.full(data_shape, True)
        elif mask.shape == data_shape:
            process_mask = mask
        elif mask.shape == data_shape[:-1]:
            process_mask = np.broadcast_to(mask[..., None], input_data_.shape)
        else:
            raise ValueError(
                f"Mask shape {mask.shape} is incompatible with input {data_shape}."
            )

        process_mask = PatchedArray(
            process_mask, p_s, p_o, padding_mode="constant", constant_values=0
        )

        center_pos = tuple(p // 2 for p in p_s)
        patch_space_size = np.prod(p_s[:-1])
        # select only queue index where process_mask is valid.
        get_it = np.zeros(input_data_.n_patches, dtype=bool)

        for i in range(len(get_it)):
            pm = process_mask.get_patch(i)
            if 100 * np.sum(pm) / pm.size > mask_threshold:
                get_it[i] = True

        select_patches = np.nonzero(get_it)[0]
        del get_it

        log.info(
            f"Processing {len(select_patches)} patches out of {input_data_.n_patches}."
        )
        log.info(f"Patch shape: {p_s}, overlap: {p_o}.")
        if progbar is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                progbar = tqdm(total=len(select_patches))
        elif progbar is not False:
            progbar.reset(total=len(select_patches))

        for i in select_patches:
            input_patch_casorati = input_data_.get_patch(i).reshape(
                patch_space_size, -1
            )
            p_denoise, maxidx, noise_var = self._patch_processing(
                input_patch_casorati,
                patch_idx=i,
                **self.input_denoising_kwargs,
            )

            p_denoise = np.reshape(p_denoise, p_s)
            rank_map.add2patch(i, maxidx)
            noise_std_estimate.add2patch(i, noise_var)
            patch_counts.add2patch(i, 1)

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
        log.info("Finished processing patches.")
        output_data = output_data._arr
        patch_weights = patch_weights._arr
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.recombination in ["average", "weighted"]:
                output_data /= patch_weights

            noise_std_estimate = noise_std_estimate._arr / patch_counts._arr
            rank_map = rank_map._arr / patch_counts._arr

        noise_std_estimate[~process_mask._arr] = 0
        output_data[~process_mask._arr] = 0
        rank_map[~process_mask._arr] = 0

        return output_data, patch_weights, noise_std_estimate, rank_map

    def _get_patch_param(
        self, data_shape: tuple[int, int, int, int]
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        """Return tuple for patch_shape and patch_overlap.

        It works from whatever the  input format was (int or list).
        This method also ensure that the patch will provide tall and skinny matrices.
        """
        return _patch_param(self.p_shape, data_shape), _patch_param(
            self.p_ovl, data_shape
        )


def _patch_param(
    patch_param: Any, data_shape: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Return tuple for patch_param.

    It works from whatever the  input format was (int or list).
    This method also ensure that the patch will provide tall and skinny matrices.
    """
    if isinstance(patch_param, int):
        patch_param = (patch_param,) * len(data_shape)
    elif isinstance(patch_param, tuple | list):
        if len(patch_param) == 1:
            patch_param = (patch_param[0],) * (len(data_shape) - 1) + (data_shape[-1],)
        elif len(patch_param) == len(data_shape) - 1:
            patch_param = tuple(patch_param) + (data_shape[-1],)
        elif len(patch_param) == len(data_shape):
            patch_param = tuple(patch_param)
        else:
            raise ValueError(
                f"patch_param must have the same number of dimensions as data_shape. "
                f"Got {len(patch_param)} and {len(data_shape)}."
            )
    else:
        raise ValueError(
            f"patch_param must be an int or a sequence. Got {type(patch_param)}."
        )

    for i, (p, s) in enumerate(zip(patch_param, data_shape)):
        if p > s:
            log.warning(
                f"patch_param[{i}] is larger than data_shape[{i}]. "
                f"Reducing patch_param[{i}] from {p} to {s}."
            )
            patch_param = patch_param[:i] + (s,) + patch_param[i + 1 :]
        if p < 0:
            log.warning(f"patch_param[{i}]<1 using the data_shape[{i}]. ")
            patch_param = patch_param[:i] + (s,) + patch_param[i + 1 :]
    # Ensure patch size is not larger than data size along each axis
    patch_param = tuple(min(p, s) for p, s in zip(patch_param, data_shape))

    return patch_param  # type: ignore
