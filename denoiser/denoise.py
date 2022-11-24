"""Provides a functional entry point for denoising methods."""

from denoiser._docs import fill_doc
from denoiser.space_time.lowrank import (
    AdaptiveDenoiser,
    HybridPCADenoiser,
    MPPCADenoiser,
    NordicDenoiser,
    OptimalSVDDenoiser,
    RawSVDDenoiser,
)


def mp_pca(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    mask_threshold=50,
    recombination="weighted",
    threshold_scale=1.0,
):
    """
    Marshenko-Pastur PCA denoising method.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
        will be processed.
    mask_threshold: int
        percentage of the path that has to be in the mask so that the patch is processed.
        if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
        the voxels of the patch needs to be in the mask
    recombination: str
        The recombination method of the patch. "weighted", "average" or "center"
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
    Follows implementation of [1]_ and the one available in dipy.

    References
    ----------
    .. [1] Manjón, José V., Pierrick Coupé, Luis Concha, Antonio Buades,
           D. Louis Collins, and Montserrat Robles.
           “Diffusion Weighted Image Denoising Using Overcomplete Local PCA.”
           PLOS ONE 8, no. 9 (September 3, 2013): e73021.
           https://doi.org/10.1371/journal.pone.0073021.

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
    return denoiser.denoise(volume_sequence, mask=mask, mask_threshold=mask_threshold)


def hybrid_pca(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    mask_threshold=50,
    noise_std=1.0,
    recombination="weighted",
):
    """
    Hybrid PCA denoising method.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
        will be processed.
    mask_threshold: int
        percentage of the path that has to be in the mask so that the patch is processed.
        if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
        the voxels of the patch needs to be in the mask
    recombination: str
        The recombination method of the patch. "weighted", "average" or "center"
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
    Follows implementation of [1]_ .

    References
    ----------
    .. [1] https://submissions.mirasmart.com/ISMRM2022/Itinerary/Files/PDFFiles/2688.html

    See Also
    --------
    denoiser.space_time.lowrank.HybridPCADenoiser
    """
    denoiser = HybridPCADenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
    )
    return denoiser.denoise(
        volume_sequence, mask=mask, mask_threshold=mask_threshold, noise_std=noise_std
    )


def raw_svt(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask_threshold=50,
    mask=None,
    threshold=1.0,
    recombination="weighted",
):
    """
    Raw singular value thresholding.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
        will be processed.
    mask_threshold: int
        percentage of the path that has to be in the mask so that the patch is processed.
        if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
        the voxels of the patch needs to be in the mask
    recombination: str
        The recombination method of the patch. "weighted", "average" or "center"
    threshold: float
        threshold use for singular value hard thresholding.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    Notes
    -----
    Simple raw hard thresholding of singular values.
    TODO: add support for soft thresholding.

    See Also
    --------
    denoiser.space_time.lowrank.MPPCADenoiser
    """
    denoiser = RawSVDDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        threshold_value=threshold,
    )
    return denoiser.denoise(
        volume_sequence, mask=mask, mask_threshold=mask_threshold, threshold_scale=1.0
    )


def nordic(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask_threshold=50,
    mask=None,
    noise_std=1.0,
    recombination="weighted",
    n_iter_threshold=10,
):
    """
    NORDIC denoising method.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
        will be processed.
    mask_threshold: int
        percentage of the path that has to be in the mask so that the patch is processed.
        if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
        the voxels of the patch needs to be in the mask
    recombination: str
        The recombination method of the patch. "weighted", "average" or "center"
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
    Follows implementation of [1]_

    References
    ----------
    .. [1] Moeller, Steen, Pramod Kumar Pisharady, Sudhir Ramanna, Christophe Lenglet,
           Xiaoping Wu, Logan Dowdle, Essa Yacoub, Kamil Uğurbil, and Mehmet Akçakaya.
           “NOise Reduction with DIstribution Corrected (NORDIC) PCA in DMRI with
           Complex-Valued Parameter-Free Locally Low-Rank Processing.”
           NeuroImage 226 (February 1, 2021): 117539.
           https://doi.org/10.1016/j.neuroimage.2020.117539.

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
        mask_threshold=mask_threshold,
        noise_std=noise_std,
        n_iter_threshold=n_iter_threshold,
    )


def optimal_thresholding(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    mask_threshold=50,
    loss="fro",
    noise_std=None,
    recombination="weighted",
    eps_marshenko_pastur=1e-7,
):
    """
    Optimal thresholing denoising method.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
        will be processed.
    mask_threshold: int
        percentage of the path that has to be in the mask so that the patch is processed.
        if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
        the voxels of the patch needs to be in the mask
    loss: str
        The loss for which the optimal thresholding is perform.
    noise_std: an estimation of the spatial noise map standard deviation.
        If None, the noise map is estimated using the Marcenko-Pastur distribution.
    recombination: str
        The recombination method of the patch. "weighted", "average" or "center"
    eps_marshenko_pastur: float
        The precision with which the optimal threshold is computed.

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    Notes
    -----
    Reimplement in python  [1]_

    References
    ----------
    .. [1] Gavish, Matan, and David L. Donoho. “Optimal Shrinkage of Singular Values.”
        IEEE Transactions on Information Theory 63, no. 4 (April 2017): 2137–52.
        https://doi.org/10.1109/TIT.2017.2653801.


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
        volume_sequence,
        mask=mask,
        noise_std=noise_std,
        mask_threshold=mask_threshold,
        eps_marshenko_pastur=eps_marshenko_pastur,
    )


def adaptive_thresholding(
    volume_sequence,
    patch_shape,
    patch_overlap,
    mask=None,
    mask_threshold=50,
    recombination="weighted",
    method="SURE",
    nbsim=500,
    tau0=None,
    gamma0=None,
    noise_std=1.0,
):
    """
    Optimal thresholing denoising method.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        The volume shape to denoise
    patch_shape: tuple
        The patch shape
    patch_overlap: tuple
        the overlap of each pixel
    mask: numpy.ndarray
        A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
        will be processed.
    mask_threshold: int
        percentage of the path that has to be in the mask so that the patch is processed.
        if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
        the voxels of the patch needs to be in the mask
    recombination: str
        The recombination method of the patch. "weighted", "average" or "center"

    Returns
    -------
    tuple
        numpy.ndarray: The denoised sequence of volume
        numpy.ndarray: The weight of each pixel after the processing.
        numpy.ndarray: If possible, the noise variance distribution in the volume.

    Notes
    -----
    Adapt the R package presented in  [1]_

    References
    ----------
    .. [1] J. Josse and S. Sardy, “Adaptive Shrinkage of singular values.”
           arXiv, Nov. 22, 2014.
           doi: 10.48550/arXiv.1310.6602.

    See Also
    --------
    denoiser.space_time.lowrank.OptimalSVDDenoiser
    """
    denoiser = AdaptiveDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        method=method,
        nbsim=nbsim,
    )
    return denoiser.denoise(
        volume_sequence, mask, mask_threshold, tau0, noise_std, gamma0
    )


def volume_denoise(volume_sequence, method):
    """Denoise each volume independently.

    Parameters
    ----------
    volume_sequence: numpy.ndarray
        a sequence of volume to denoise
    method: str
        The denoising method to use. Available will be
        - `"bm3d"`
        - `"dip"`

    """
    ...
