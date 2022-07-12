import numpy as np

NIBABEL_AVAILABLE = True
try:
    import nibabel as nib
except ImportError:
    NIBABEL_AVAILABLE = False


def load_complex_nifti(mag_file, phase_file, filename=None):
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
