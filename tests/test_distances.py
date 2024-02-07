import pytest
import numpy as np
from distclassipy.distances import Distance

# Initialize the Distance class to use its methods for testing
distance = Distance()


# Test for the accuracy distance calculation
def test_acc_distance():
    # Define two sample vectors
    u = np.array([1, 2, 3])
    v = np.array([2, 4, 6])
    # Calculate the expected result manually
    expected = np.mean([np.sum(np.abs(u - v)), np.max(np.abs(u - v))])
    # Assert that the calculated distance matches the expected result
    assert distance.acc(u, v) == expected


# Test for the Vicis Wave Hedges distance calculation
def test_vicis_wave_hedges():
    # Define two sample vectors
    u = np.array([1, 2, 3])
    v = np.array([2, 4, 6])
    # Calculate the minimum of u and v element-wise
    uvmin = np.minimum(u, v)
    # Calculate the absolute difference between u and v
    u_v = np.abs(u - v)
    # Calculate the expected result manually
    expected = np.sum(np.where(uvmin != 0, u_v / uvmin, 0))
    # Assert that the calculated distance matches the expected result
    assert distance.vicis_wave_hedges(u, v) == expected


# HAVE TO ADD MORE TESTS
