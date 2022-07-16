import numpy as np

NIBABEL_AVAILABLE = True
try:
    import nibabel as nib
except ImportError:
    NIBABEL_AVAILABLE = False


def load_complex_nifti(mag_file, phase_file, filename=None): # pragma: no cover
    """Load two nifti image (magnitude and phase) to create a complex valued array.

    Optionally, the result can be save as a .npy file

    Parameters
    ----------
    mag_file: str
        The source magnitude file
    phase_file: str
        The source phase file
    filename: str, default None
        The output filename
    """
    if not NIBABEL_AVAILABLE:
        raise RuntimeError(
            "nibabel is not available, please install it to load experimental data"
        )

    mag = nib.load(mag_file).get_fdata()
    phase = nib.load(phase_file).get_fdata()
    print(np.min(phase), np.max(phase))
    print(np.min(mag), np.max(mag))
    img = mag * np.exp(1j * phase)

    if filename is not None:
        np.save(filename, img)
    return img



def _zigzag(rows, columns):
    """Return a list of coordinate in zigzag pattern"""
    pattern=[[] for i in range(rows+columns-1)]
    for i in range(rows):
        for j in range(columns):
            s=i+j
            if(s%2 ==0):
                pattern[s].insert(0,(i,j))
            else:
                pattern[s].append((i,j))

    return [p for pp in pattern for p in pp]

def array2zigzag(array):
    """Flatten a 2D array with a zigzag pattern.

    Parameters
    ----------
    array: numpy.ndarray
        a 2 Array to flatten.

    Returns
    -------
    numpy.ndarray
        A flattened 1D array.

    Notes
    -----
    Flattenning a 2D array with a zigzag pattern preserves a spatial contiguity
    in the flattened version, unlike the classical row or column-wise stacking.
    """
    if array.ndim != 2:
        raise ValueError("The array shoud be 2-dimensional.")
    zigzag = _zigzag(*array.shape)

    return array[tuple(zip(*zigzag))]

def zigzag2array(array, shape):
    """
    Reshape a 1d array into a 2D shape following a zigzag pattern.

    Parameters
    ----------
    array: numpy.ndarray
        A 1D array.
    shape: tuple
        The row and column size of the new array

    Returns
    -------
    numpy.ndarray
        The 2D array
    """

    if len(shape) != 2:
        raise ValueError("The shape of the new array should be 2-dimensional")
    zigzag = _zigzag(*shape)

    new_array = np.zeros(shape, dtype=array.dtype)

    for idx, z in  enumerate(zigzag):
        print(idx, z)
        new_array[z[0], z[1]] = array[idx]
    return new_array
