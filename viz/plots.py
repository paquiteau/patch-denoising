import matplotlib.pyplot as plt


def carpet_plot(arr, unfold="classic", colorbar=True, transpose=False):
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
    """
    ...
