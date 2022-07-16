"""
This module provides a functional entry point for denoising methods.
"""

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


def optimal_thresholding(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    loss="fro",
    recombination="weighted",
    eps_marshenko_pastur=1e-7,
):
    """
    Optimal thresholing denoising method

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
    loss: str
        The loss for which the optimal thresholding is perform.
    recombination: str
        The recombination method of the patch. "weighted" or "mean"
    eps_marshenko_pastur: float
        The precision with which the optimal threshold is computed.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    See Also
    --------
    denoiser.space_time.lowrank.OptimalSVDDenoiser
    """
    denoiser = OptimalSVDDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        loss=loss,
    )
    return denoiser.denoise(
        volume_sequence, mask=mask, eps_marshenko_pastur=eps_marshenko_pastur
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
