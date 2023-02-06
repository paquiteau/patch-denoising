"""Provides a functional entry point for denoising methods."""
from ._docs import fill_doc
from .space_time.lowrank import (
    AdaptiveDenoiser,
    HybridPCADenoiser,
    MPPCADenoiser,
    NordicDenoiser,
    OptimalSVDDenoiser,
    RawSVDDenoiser,
)


@fill_doc
def mp_pca(
    input_data,
    patch_shape,
    patch_overlap,
    mask=None,
    mask_threshold=50,
    recombination="weighted",
    threshold_scale=1.0,
    progbar=None,
):
    """
    Marshenko-Pastur PCA denoising method.

    Parameters
    ----------
    $standard_config
    threshold_scale: float
        An extra factor for the patch denoising.

    Returns
    -------
    $denoise_return

    Notes
    -----
    Follows implementation of [#]_ and the one available in dipy.

    References
    ----------
    .. [#] Manjón, José V., Pierrick Coupé, Luis Concha, Antonio Buades,
           D. Louis Collins, and Montserrat Robles.
           “Diffusion Weighted Image Denoising Using Overcomplete Local PCA.”
           PLOS ONE 8, no. 9 (September 3, 2013): e73021.
           https://doi.org/10.1371/journal.pone.0073021.

    See Also
    --------
    patch_denoise.space_time.lowrank.MPPCADenoiser
    """
    denoiser = MPPCADenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        threshold_scale=threshold_scale,
    )
    return denoiser.denoise(input_data, mask=mask, mask_threshold=mask_threshold)


@fill_doc
def hybrid_pca(
    input_data,
    patch_shape,
    patch_overlap,
    mask=None,
    mask_threshold=50,
    noise_std=1.0,
    recombination="weighted",
    progbar=None,
):
    """
    Hybrid PCA denoising method.

    Parameters
    ----------
    $standard_config
    $noise_std

    Returns
    -------
    $denoise_return

    Notes
    -----
    Follows implementation of [#]_ .

    References
    ----------
    .. [#]
    https://submissions.mirasmart.com/ISMRM2022/Itinerary/Files/PDFFiles/2688.html

    See Also
    --------
    patch_denoise.space_time.lowrank.HybridPCADenoiser
    """
    denoiser = HybridPCADenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
    )
    return denoiser.denoise(
        input_data,
        mask=mask,
        mask_threshold=mask_threshold,
        noise_std=noise_std,
        progbar=progbar,
    )


@fill_doc
def raw_svt(
    input_data,
    patch_shape,
    patch_overlap,
    mask_threshold=50,
    mask=None,
    threshold=1.0,
    recombination="weighted",
    progbar=None,
):
    """
    Raw singular value thresholding.

    Parameters
    ----------
    $standard_config
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
    patch_denoise.space_time.lowrank.MPPCADenoiser
    """
    denoiser = RawSVDDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        threshold_value=threshold,
    )
    return denoiser.denoise(
        input_data,
        mask=mask,
        mask_threshold=mask_threshold,
        threshold_scale=1.0,
        progbar=progbar,
    )


@fill_doc
def nordic(
    input_data,
    patch_shape,
    patch_overlap,
    mask_threshold=50,
    mask=None,
    noise_std=1.0,
    recombination="weighted",
    n_iter_threshold=10,
    progbar=None,
):
    """
    NORDIC denoising method.

    Parameters
    ----------
    $standard_config
    $noise_std
    n_iter_threshold: int
        The number of Monte-Carlo Simulation to estimate the global threshold.

    Returns
    -------
    $denoise_return

    Notes
    -----
    Follows implementation of [#]_

    References
    ----------
    .. [#] Moeller, Steen, Pramod Kumar Pisharady, Sudhir Ramanna, Christophe Lenglet,
           Xiaoping Wu, Logan Dowdle, Essa Yacoub, Kamil Uğurbil, and Mehmet Akçakaya.
           “NOise Reduction with DIstribution Corrected (NORDIC) PCA in DMRI with
           Complex-Valued Parameter-Free Locally Low-Rank Processing.”
           NeuroImage 226 (February 1, 2021): 117539.
           https://doi.org/10.1016/j.neuroimage.2020.117539.

    See Also
    --------
    patch_denoise.space_time.lowrank.NordicDenoiser
    """
    denoiser = NordicDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
    )
    return denoiser.denoise(
        input_data,
        mask=mask,
        mask_threshold=mask_threshold,
        noise_std=noise_std,
        n_iter_threshold=n_iter_threshold,
        progbar=progbar,
    )


@fill_doc
def optimal_thresholding(
    input_data,
    patch_shape,
    patch_overlap,
    mask=None,
    mask_threshold=50,
    loss="fro",
    noise_std=None,
    recombination="weighted",
    eps_marshenko_pastur=1e-7,
    progbar=None,
):
    """
    Optimal thresholing denoising method.

    Parameters
    ----------
    $standard_config
    $noise_std
    loss: str
        The loss for which the optimal thresholding is perform.
    eps_marshenko_pastur: float
        The precision with which the optimal threshold is computed.

    Returns
    -------
    $denoise_return

    Notes
    -----
    Reimplement of the original Matlab code [#]_ in python.

    References
    ----------
    .. [#] Gavish, Matan, and David L. Donoho. “Optimal Shrinkage of Singular Values.”
        IEEE Transactions on Information Theory 63, no. 4 (April 2017): 2137–52.
        https://doi.org/10.1109/TIT.2017.2653801.


    See Also
    --------
    patch_denoise.space_time.lowrank.OptimalSVDDenoiser
    """
    denoiser = OptimalSVDDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        loss=loss,
    )
    return denoiser.denoise(
        input_data,
        mask=mask,
        noise_std=noise_std,
        mask_threshold=mask_threshold,
        eps_marshenko_pastur=eps_marshenko_pastur,
        progbar=progbar,
    )


@fill_doc
def adaptive_thresholding(
    input_data,
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
    progbar=None,
):
    """
    Optimal thresholing denoising method.

    Parameters
    ----------
    $input_config
    $noise_std
    method: str
        The adaptive method to use "SURE" or "GSURE"
    nbsim:
        Number of simulation for computing sure estimator
    tau:
        Simulation parameter.
    gamma0:
        Simulation parameter.

    Returns
    -------
    $denoise_return

    Notes
    -----
    Reimplements the R package [#]_ in python.

    References
    ----------
    .. [#] J. Josse and S. Sardy, “Adaptive Shrinkage of singular values.”
           arXiv, Nov. 22, 2014.
           doi: 10.48550/arXiv.1310.6602.

    See Also
    --------
    patch_denoise.space_time.AdaptiveDenoiser
    """
    denoiser = AdaptiveDenoiser(
        patch_shape,
        patch_overlap,
        recombination=recombination,
        method=method,
        nbsim=nbsim,
    )
    return denoiser.denoise(
        input_data,
        mask,
        mask_threshold,
        tau0,
        noise_std,
        gamma0,
        progbar=progbar,
    )
