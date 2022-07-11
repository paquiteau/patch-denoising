from denoiser.space_time.lowrank import (
    MPPCADenoiser,
    HybridPCADenoiser,
    OptimalSVDDenoiser,
    NordicDenoiser,
    RawSVDDenoiser,
)


_DENOISER = {
    "mp-pca": MPPCADenoiser,
    "hybrid-pca": HybridPCADenoiser,
    "nordic": NordicDenoiser,
    "optimal": OptimalSVDDenoiser,
    "raw": RawSVDDenoiser,
}


def local_lowrank_denoise(
    volume_sequence,
    method,
    patch_shape,
    patch_overlap,
    mask=None,
    noise_std=None,
    **kwargs,
):
    """
    Generic function to perform a local low-rank based denoising.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The noisy volume sequence
    method: str
        The denoising method to use. available are:
        - `"mppca"`
        - `"hybridpca"`
        - `"nordic"`
        = `"raw"`
        - `"optimal-fro"`
        - `"optimal-nuc"`
        - `"optimal-op"`

    patch_shape: tuple
        The shape of the patch to denoise. The number of pixel in a patch should be
        larger than the number of frame
    patch_overlap: tuple
        The number of voxel which should overlap between two patches along each dimensions.
    noise_std: float or numpy.npdarray
        Extra noise parameter estimation, used for hybrid-pca and nordic
    **kwargs
        Extra parameters for the denosing methods.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    See Also
    --------
    denoiser.space_time.lowrank
        Module of the implementation of the low rank denoising methods.
    mp_pca
        Function to perform the mp-pca denoising
    hybrid_pca
        Function to perform the Hybrid-pca
    nordic
        Function to perform the NORDIC denoising
    """

    if "optimal" in method:
        denoiser_klass = _DENOISER["optimal"]
    else:
        try:
            denoiser_klass = _DENOISER[method]
        except KeyError as e:
            raise ValueError(f"unknown method name `{method}`") from e

    denoiser = denoiser_klass(patch_shape, patch_overlap, **kwargs)

    denoised_sequence, patch_weight, noise_std_estimate = denoiser.denoiser(
        volume_sequence, noise_std=noise_std, mask=mask
    )

    return denoised_sequence, patch_weight, noise_std_estimate


def mp_pca(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    recombination="weighted",
    threshold_scale=1.0,
):
    """
    Marshenko-Pastur PCA denoising method

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROi will be processed.
    recombination: str
        The recombination method of the patch. "weighted" or "mean"
    threshold_scale: float
        An extra factor for the patch denoising.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    Notes
    -----


    See Also
    --------
    denoiser.space_time.lowrank.MPPCADenoiser
    """
    denoiser = MPPCADenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        threshold_scale=threshold_scale,
    )
    return denoiser.denoise(volume_sequence, mask=mask)


def hybrid_pca(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    noise_std=1.0,
    recombination="weighted",
):
    """
    Hybrid PCA denoising method

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROi will be processed.
    recombination: str
        The recombination method of the patch. "weighted" or "mean"
    threshold_scale: float
        An extra factor for the patch denoising.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    Notes
    -----


    See Also
    --------
    denoiser.space_time.lowrank.HybridPCADenoiser
    """
    denoiser = HybridPCADenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
    )
    return denoiser.denoise(volume_sequence, mask=mask, noise_std=noise_std)


def raw_svt(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    noise_std=1.0,
    recombination="weighted",
):
    """
    Raw singular value thresholding

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROi will be processed.
    recombination: str
        The recombination method of the patch. "weighted" or "mean"
    noise_std: float or numpy.npdarray
        An estimation of the noise standard deviation.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    Notes
    -----


    See Also
    --------
    denoiser.space_time.lowrank.MPPCADenoiser
    """
    denoiser = HybridPCADenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
    )
    return denoiser.denoise(volume_sequence, mask=mask, noise_std=noise_std)


def nordic(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    noise_std=1.0,
    recombination="weighted",
    n_iter_threshold=10,
):
    """
    NORDIC denoising method

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROi will be processed.
    recombination: str
        The recombination method of the patch. "weighted" or "mean"
    threshold_scale: float
        An extra factor for the patch denoising.
    n_iter_threshold: int
        The number of Monte-Carlo Simulation to estimate the global threshold.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    Notes
    -----


    See Also
    --------
    denoiser.space_time.lowrank.NordicDenoiser
    """
    denoiser = NordicDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
    )
    return denoiser.denoise(
        volume_sequence,
        mask=mask,
        noise_std=noise_std,
        n_iter_threshold=n_iter_threshold,
    )


def volume_denoise(volume_sequence, method):
    """Denoise each volume independently.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        a sequence of volume to denoise
    method: str
        The denoising method to use. Available are:
        - `"bm3d"`
        - `"dip"`

    """
    ...
