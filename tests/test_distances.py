import pytest
import numpy as np
from distclassipy.distances import Distance

# Initialize the Distance class to use its methods for testing
distance = Distance()


def test_all_distances():
    # Define two sample vectors
    u = np.array([1, 2, 3])
    v = np.array([1, 2, 3])
    for func_name in dir(distance):
        if callable(getattr(distance, func_name)) and not func_name.startswith("__"):
            func = getattr(distance, func_name)
            d = func(u, v)
            assert d >= 0
