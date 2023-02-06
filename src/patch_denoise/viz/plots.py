"""Plottings function for Spatio-temporal data."""
import matplotlib.pyplot as plt
import numpy as np

from .utils import array2zigzag


def carpet_plot(
    arr,
    unfold="classic",
    colorbar=True,
    transpose=False,
    figsize=None,
):  # pragma: no cover
    """Carpet Plot of ND array.

    The last dimension if plotted on the horizontal axis,
    The other are flattened on the vertical one.

    Parameters
    ----------
    arr: numpy.ndarray
        The  ND array to plot.
    unfold: str
        The method to unfold the (N-1)D first dimensions.
        If "classic" the data is simply row-wise unfolded
        If "zigzag" (only for 2D spatial dimensions), the data is unfolded following
        a diagonal zig-zag pattern starting from top left.
    colorbar: bool
        flag to show a colorbar
    transpose: bool
        If true, switch the horizontal and vertical axis.

    Returns
    -------
    matplotlib.pyplot.figure:
        The plotted figure.
    """
    if unfold == "classic":
        unfolded_arr = np.reshape(arr, (-1, arr.shape[-1]))
    elif unfold == "zigzag":
        if arr.ndim != 3:
            raise ValueError(
                "input array should be 3 dimensional, "
                "with temporal dimension in last position."
            )
        unfolded_arr = array2zigzag(arr)
    else:
        raise ValueError(f"invalid unfold method {unfold}")

    if np.iscomplexobj(unfolded_arr):
        unfolded_arr = np.abs(unfolded_arr)

    fig, ax = plt.subplots(figsize=figsize)

    if transpose:
        im = ax.imshow(unfolded_arr.T)
        ax.set_ylabel("time")
        ax.set_xlabel("voxel")
    else:
        im = ax.imshow(unfolded_arr)
        ax.set_xlabel("time")
        ax.set_ylabel("voxel")
    if colorbar:
        fig.colorbar(im)
    return fig
