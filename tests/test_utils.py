import pytest

from patch_denoise.viz.utils import zigzag2array, array2zigzag, _zigzag
import numpy as np
from numpy.testing import assert_equal


@pytest.mark.parametrize("rows, columns", [(10, 10), (21, 11), (11, 21)])
def test_zigzag(rows, columns):
    arr = np.arange(rows * columns).reshape(rows, columns)

    zigzag = _zigzag(*arr.shape)
    print(tuple(zip(*zigzag)))
    arr_new = zigzag2array(array2zigzag(arr), (rows, columns))

    assert_equal(arr_new, arr)
