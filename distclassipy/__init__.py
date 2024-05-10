"""
A module for using distance metrics for classification.

Classes:
    DistanceMetricClassifier - A classifier that uses a specified distance metric for classification.
    Distance - A class that provides various distance metrics for use in classification.
"""

from .classifier import DistanceMetricClassifier  # noqa
from .distances import Distance  # noqa

__version__ = "0.1.4"
