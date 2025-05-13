"""A module for using distance metrics for classification.

Classes:
    DistanceMetricClassifier - A classifier that uses a specified distance metric for
                               classification.
    Distance - A class that provides various distance metrics for use in classification.


Copyright (C) 2024  Siddharth Chaini
-----
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .classifier import (
    DistanceMetricClassifier,
    EnsembleDistanceClassifier,
)
from .distances import _ALL_METRICS

__version__ = "0.2.2a1"

__all__ = [
    "DistanceMetricClassifier",
    "EnsembleDistanceClassifier",
    "Distance",
    "_ALL_METRICS",
]
