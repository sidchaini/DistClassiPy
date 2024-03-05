"""
A module for using distance metrics for classification.

Classes:
    DistanceMetricClassifier - A classifier that uses a specified distance metric for classification.
    Distance - A class that provides various distance metrics for use in classification.
"""

from .classifier import (
    DistanceMetricClassifier,
)  # Importing the DistanceMetricClassifier from the classifier module
from .distances import (
    Distance,
)  # Importing the Distance class from the distances module
from .version import __version__
