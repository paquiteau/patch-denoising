from itertools import product
import pytest
import numpy as np


@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(42)
