"""Tests for DistanceAnomaly behaviour not tied to a specific scaling option."""

import pytest

from distclassipy.anomaly import DistanceAnomaly


def test_unknown_constructor_argument_is_rejected():
    """A typo must fail loudly instead of being silently ignored."""
    with pytest.raises(TypeError):
        DistanceAnomaly(n_neighbours=5)  # sic: n_neighbors
