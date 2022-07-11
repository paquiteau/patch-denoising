import numpy as np


class BaseSpaceTimeDenoiser:
    """
    Base Class for Patch-based denoising methods for dynamical data.

    Parameters
    ----------
    patch_shape: tuple
    patch_overlap: tuple
    recombinaison: str

    """

    def __init__(self, patch_shape, patch_overlap, recombination="weighted"):
        self.p_shape = patch_shape
        self.p_ovl = patch_overlap
        self.recombination = recombination

        self.input_denoising_kwargs = dict()

    def denoise(self, input_data, mask=None):
        """Denoise the input_data, according to mask.

        Parameters
        ----------
        input_data: numpy.ndarray
            The input data to denoise. It should be a ND array, and the last
            dimension should a dynamically varying one (eg time).
        mask: numpy.ndarray
            A boolean array defining a ROI where the patch denoising will be process
            It should be a (N-1)D boolean array.
        denoiser_kwargs: dict
            Extra runtime parameters passed to the patch denoising method.

        Returns
        -------
        tuple:
            output_data: numpy.ndarray
                The denoised data
            patch_weight: numpy.ndarray
                The weight associated to each voxel.
            noise_std_map: numpy.ndarray
                If available, a noise std estimation for each voxel.

        """
        data_shape = input_data.shape

        # Create Default mask
        if mask is None:
            mask = np.full(data_shape[:-1], True)
        patch_shape, patch_overlap = self.__get_patch_param(data_shape)

        if self.recombination == "center":
            patch_center = tuple(slice(ps // 2, ps // 2 + 1) for ps in patch_shape)
        patchs_weight = np.zeros(data_shape[:-1], np.float32)
        noise_std_estimate = np.zeros(data_shape[:-1], dtype=np.float32)

        for patch_tl in self.__get_patch_locs(
            patch_shape, patch_overlap, data_shape[:-1]
        ):

            patch_slice = tuple(
                slice(tl, tl + ps) for tl, ps in zip(patch_tl, patch_shape)
            )
            if not np.any(mask[patch_slice]):
                continue  # patch is outside the mask.
            # building the casoratti matrix
            patch = np.reshape(
                input_data[patch_slice],
                (-1, input_data.shape[-1]),
            )

            p_denoise, *extras = self._patch_processing(
                patch,
                patch_slice=patch_slice,
                **self.input_denoising_kwargs,
            )

            p_denoise = np.reshape(p_denoise, (*patch_shape, -1))
            if self.recombination == "center":
                patch_center_img = tuple(
                    slice(ptl + ps // 2, ptl + ps // 2 + 1)
                    for ptl, ps in zip(patch_tl, patch_shape)
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
        # Averaging the overlapping pixels.
        output_data /= patchs_weight[..., None]
        noise_std_estimate /= patchs_weight

        output_data[~mask] = 0

        return output_data, patchs_weight, noise_std_estimate

    def _patch_processing(patch, patch_slice=None, **kwargs):
        """Processing of pach"""
        raise NotImplementedError

    def __get_patch_param(self, data_shape):
        pp = (None, None)
        for i, attr in enumerate(["p_ovl", "p_shape"]):
            p = getattr(self, attr)
            if isinstance(p, [tuple, list]):
                p = tuple(self.p_ovl)
            elif isinstance(p, [int, np.integer]):
                p = (p,) * len(data_shape)
            pp[i] = p
        if np.prod(pp[1]) < data_shape[-1]:
            raise ValueError(
                "the number of voxel in patch is smaller than the last dimension,"
                + " this makes an ill-conditioned matrix for SVD.",
            )
        return pp

    def __get_patch_locs(self, p_shape, p_ovl, v_shape):

        """
        Get all the patch top-left corner locations.

        Parameters
        ----------
        vol_shape : tuple
            The volume shape
        patch_shape : tuple
            The patch shape
        patch_overlap : tuple
            The overlap of patch for each dimension.

        Returns
        -------
        numpy.ndarray
            All the patch top-left corner locations.
        """
        # Create an iterator for all the possible patches top-left corner location.
        if len(v_shape) != len(p_shape) or len(v_shape) != len(p_ovl):
            raise ValueError("Dimension mismatch between the arguments.")

        ranges = []
        for v_s, p_s, p_o in zip(v_shape, p_shape, p_ovl):
            if p_o >= p_s:
                raise ValueError(
                    "Overlap should be a non-negative integer"
                    + "smaller than patch_size",
                )
            last_idx = v_s - p_s
            range_ = np.arange(0, last_idx, p_s - p_o, dtype=np.int32)
            if range_[-1] < last_idx:
                range_ = np.append(range_, last_idx)
            ranges.append(range_)
        # fast ND-Cartesian product from https://stackoverflow.com/a/11146645
        patch_locs = np.empty(
            [len(arr) for arr in ranges] + [len(p_shape)],
            dtype=np.int32,
        )
        for idx, coords in enumerate(np.ix_(*ranges)):
            patch_locs[..., idx] = coords

        return patch_locs.reshape(-1, len(p_shape))