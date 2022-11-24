"""Docstring utils

docdict contains the standard argument documentation found across the package.


source:

https://github.com/nilearn/nilearn/blob/main/nilearn/_utils/docs.py
https://github.com/mne-tools/mne-python/blob/main/mne/utils/docs.py
"""
import sys

docdict_indented = {}
docdict = {}

docdict = {
    "patch_config": """
patch_shape: tuple
    The patch shape
patch_overlap: tuple
    the overlap of each pixel
recombination: str, optional
    The recombination method of the patch. "weighted", "average" or "center".
    default "weighted".
""",
    "mask_config": """
mask: numpy.ndarray
    A boolean array, defining a ROI in the volume. Only patch with voxels in the ROI
    will be processed.
mask_threshold: int
    percentage of the path that has to be in the mask so that the patch is processed.
    if mask_threshold = -1, all the patch are processed, if mask_threshold=100, all
    the voxels of the patch needs to be in the mask
""",
    "denoise_return": """
tuple
    numpy.ndarray: The denoised sequence of volume
    numpy.ndarray: The weight of each pixel after the processing.
    numpy.ndarray: If possible, the noise variance distribution in the volume.
    """,
    "standard_config": """
input_data: numpy.ndarray
    The input data to denoise. It should be a ND array, and the last
    dimension should a dynamically varying one (eg time).
""",
    "noise_std": """
noise_std: float or numpy.ndarray
    An estimation of the spatial noise map standard deviation.
""",
}

# complete the standard config with patch and mask configuration.
docdict["standard_config"] += docdict["patch_config"] + docdict["mask_config"]


def _indentcount_lines(lines):
    """Minimum indent for all lines in line list.

    >>> lines = [' one', '  two', '   three']
    >>> _indentcount_lines(lines)
    1
    >>> lines = []
    >>> _indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> _indentcount_lines(lines)
    1
    >>> _indentcount_lines(['    '])
    0
    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return


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
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError("Error documenting %s:\n%s" % (funcname, str(exp)))
    return f
