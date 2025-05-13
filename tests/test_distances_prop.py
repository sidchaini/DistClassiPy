import math

import distclassipy as dcpy
from distclassipy import distances

from hypothesis import given, strategies as st

import numpy as np

import pytest

# Strategy to generate arrays of floats
arrays = st.integers(min_value=1, max_value=20).flatmap(
    lambda n: st.tuples(
        st.lists(
            st.floats(
                min_value=1,
                max_value=10e3,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=n,
            max_size=n,
        ).map(np.array),
        st.lists(
            st.floats(
                min_value=1,
                max_value=10e3,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=n,
            max_size=n,
        ).map(np.array),
    )
)


@pytest.mark.parametrize(
    "metric", [m for m in dcpy._ALL_METRICS if m not in ["marylandbridge"]]
)  # Note: Maryland bridge is excluded as it fails this test.
@given(arrays)
def test_non_negative(metric, data):
    u, v = data
    assert getattr(distances, metric)(u, v) >= 0


@pytest.mark.parametrize(
    "metric", [m for m in dcpy._ALL_METRICS if m not in ["motyka"]]
)  # Note: Motyka is excluded as it fails this test.
@given(arrays)
def test_self_distance(metric, data):
    u, _ = data
    assert math.isclose(getattr(distances, metric)(u, u), 0)


@pytest.mark.parametrize("metric", dcpy._ALL_METRICS)
@given(arrays)
def test_symmetry(metric, data):
    u, v = data
    assert math.isclose(
        getattr(distances, metric)(u, v), getattr(distances, metric)(v, u)
    )
