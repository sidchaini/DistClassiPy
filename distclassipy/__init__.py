"""A module for using distance metrics for classification and anomaly detection.

Classes:
    DistanceMetricClassifier - A classifier that uses a specified distance metric for
                               classification.
    EnsembleDistanceClassifier - An ensemble classifier across distance metrics.
    DistanceAnomaly - A multi-metric distance-based anomaly detector.

Functions:
    cdist - scipy-style pairwise distances, supporting all custom metrics at C speed.


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

from ._dispatch import cdist
from .anomaly import DistanceAnomaly
from .classifier import (
    DistanceMetricClassifier,
    EnsembleDistanceClassifier,
)
from .distances import _ALL_METRICS, _UNIQUE_METRICS

__version__ = "0.4.0"

__all__ = [
    "DistanceMetricClassifier",
    "EnsembleDistanceClassifier",
    "DistanceAnomaly",
    "cdist",
    "_ALL_METRICS",
    "_UNIQUE_METRICS",
]
