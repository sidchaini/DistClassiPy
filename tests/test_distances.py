# import pytest
from distclassipy.distances import Distance

import numpy as np

from hypothesis import given, strategies as st

# Initialize the Distance class to use its methods for testing
distance = Distance()

# Strategy to generate arrays of floats
arrays = st.integers(min_value=1, max_value=20).flatmap(
    lambda n: st.tuples(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=n,
            max_size=n,
        ).map(np.array),
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=n,
            max_size=n,
        ).map(np.array),
    )
)


@given(arrays)
def test_euclidean_non_negative(data):
    u, v = data
    assert distance.euclidean(u, v) >= 0


@given(arrays)
def test_manhattan_non_negative(data):
    u, v = data
    assert distance.cityblock(u, v) >= 0


@given(arrays)
def test_chebyshev_non_negative(data):
    u, v = data
    assert distance.chebyshev(u, v) >= 0


@given(arrays)
def test_euclidean_self_distance(data):
    u, _ = data
    assert distance.euclidean(u, u) == 0


@given(arrays)
def test_manhattan_self_distance(data):
    u, _ = data
    assert distance.cityblock(u, u) == 0


@given(arrays)
def test_chebyshev_self_distance(data):
    u, _ = data
    assert distance.chebyshev(u, u) == 0


@given(arrays)
def test_euclidean_symmetry(data):
    u, v = data
    assert distance.euclidean(u, v) == distance.euclidean(v, u)


@given(arrays)
def test_manhattan_symmetry(data):
    u, v = data
    assert distance.cityblock(u, v) == distance.cityblock(v, u)


@given(arrays)
def test_chebyshev_symmetry(data):
    u, v = data
    assert distance.chebyshev(u, v) == distance.chebyshev(v, u)


# Run the tests
if __name__ == "__main__":
    test_euclidean_non_negative()
    test_manhattan_non_negative()
    test_chebyshev_non_negative()
    test_euclidean_self_distance()
    test_manhattan_self_distance()
    test_chebyshev_self_distance()
    test_euclidean_symmetry()
    test_manhattan_symmetry()
    test_chebyshev_symmetry()
