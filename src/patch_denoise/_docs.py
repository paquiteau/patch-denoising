"""Docstring utils.

docdict contains the standard argument documentation found across the package.

Docstring can then use templated argument such as ``$patch_config`` that will be
substitute by their definition (see docdict items).

source:
"""

import inspect
from string import Template


docdict = dict(
    patch_config="""
patch_shape: tuple
    The patch shape
patch_overlap: tuple
    the overlap of each pixel
recombination: str, optional
    The recombination method of the patch. "weighted", "average" or "center".
    default "weighted".""",
    mask_config="""
mask: numpy.ndarray
    A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
    will be processed.
mask_threshold: int
    percentage of the path that has to be in the mask so that the patch is processed.
    if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
    the voxels of the patch needs to be in the mask""",
    denoise_return="""
tuple
    numpy.ndarray: The denoised sequence of volume
    numpy.ndarray: The weight of each pixel after the processing.
    numpy.ndarray: If possible, the noise variance distribution in the volume.""",
    input_config="""
input_data: numpy.ndarray
    The input data to denoise. It should be a ND array, and the last
    dimension should a dynamically varying one (eg time).
progbar: tqdm.tqdm Progress bar, optiononal
    An existing Progressbar, default (None) will create a new one.
    """,
    noise_std="""
noise_std: float or numpy.ndarray
    An estimation of the spatial noise map standard deviation.""",
)

# complete the standard config with patch and mask configuration.
docdict["standard_config"] = (
    docdict["input_config"] + docdict["patch_config"] + docdict["mask_config"]
)


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    try:
        docstring = Template(inspect.cleandoc(docstring)).safe_substitute(docdict)
        # remove possible gap between headline and body.
        f.__doc__ = docstring.replace("---\n\n", "---\n")
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        raise RuntimeError(f"Error documenting {funcname}s:\n{str(exp)}") from exp
    return f
