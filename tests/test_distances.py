from distclassipy.distances import Distance

import numpy as np

import pytest

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

# List all metrics
all_metrics = [
    "euclidean",
    "braycurtis",
    "canberra",
    "cityblock",
    "chebyshev",
    "clark",
    "correlation",
    "cosine",
    "hellinger",
    "jaccard",
    "lorentzian",
    "marylandbridge",
    "meehl",
    "motyka",
    "soergel",
    "wave_hedges",
    "kulczynski",
    "add_chisq",
]


@pytest.mark.parametrize("metric", all_metrics)
@given(arrays)
def test_non_negative(metric, data):
    u, v = data
    assert getattr(distance, metric)(u, v) >= 0


@pytest.mark.parametrize("metric", all_metrics)
@given(arrays)
def test_self_distance(metric, data):
    u, _ = data
    assert getattr(distance, metric)(u, u) == 0


@pytest.mark.parametrize("metric", all_metrics)
@given(arrays)
def test_symmetry(metric, data):
    u, v = data
    assert getattr(distance, metric)(u, v) == getattr(distance, metric)(v, u)
