from distclassipy.classifier import (
    DistanceMetricClassifier,
    EnsembleDistanceClassifier,
)

import numpy as np

import pytest

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator


# Test initialization of the classifier with specific parameters
def test_init():
    clf = DistanceMetricClassifier(scale=True)
    assert clf.scale is True


def test_sklearn_compatibility():
    check_estimator(DistanceMetricClassifier())


# Test fitting the classifier to a dataset
def test_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier()
    clf.fit(X, y)
    assert clf.is_fitted_ is True
    assert clf.n_features_in_ == 2


# Test making predictions with the classifier: pass metric during predict
def test_dcpy():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier()
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)


# Test fitting and predicting without scaling std
def test_predict_without_stdscale():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(scale=False)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert clf.scale is False
    assert len(predictions) == len(y)


# Test using different distance metrics - from scipy
def test_metric_scipy():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier()
    clf.fit(X, y)
    clf.predict(X, metric="cityblock")
    pass


# Test using different distance metrics - from distclassipy
def test_metric_pred():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier()
    clf.fit(X, y)
    clf.predict(X, metric="soergel")
    pass


# Test passing a distance metric during init, not predict
def test_metric_initpass():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(metric="soergel")
    clf.fit(X, y)
    clf.predict(X)
    pass


# Test passing a distance metric during init, and override during predict
def test_metric_initpass_override():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(metric="soergel")
    clf.fit(X, y)
    clf.predict(X)
    clf.predict(X, metric="canberra")
    pass


# Test using custom defined metric
def test_metric_custom():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values

    def metric_euc(u, v):
        return np.sqrt(np.sum((u - v) ** 2))

    clf = DistanceMetricClassifier()
    clf.fit(X, y)
    clf.predict(X, metric=metric_euc)
    pass


# Test using invalid metric
def test_metric_invalid():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values

    with pytest.raises(ValueError):
        clf = DistanceMetricClassifier()
        clf.fit(X, y)
        clf.predict(X, metric="chaini")


# Test setting central statistical method to median
def test_central_stat_median():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(central_stat="median", dispersion_stat="iqr")
    clf.fit(X, y)
    assert clf.central_stat == "median"


# Test setting central statistical method to mean
def test_central_stat_mean():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(central_stat="mean")
    clf.fit(X, y)
    assert clf.central_stat == "mean"


# Test prediction error when the classifier is not fitted
def test_predict_without_fit():
    clf = DistanceMetricClassifier()
    with pytest.raises(ValueError):
        clf.predict(np.array([[1, 2]]))


# Test confidence calculation error when analysis is not performed
def test_calculate_confidence_without_analysis():
    clf = DistanceMetricClassifier()
    with pytest.raises(ValueError):
        clf.calculate_confidence()


# Test confidence calculation
def test_confidence_calculation():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier()
    clf.fit(X, y)
    clf.predict_and_analyse(X)
    distance_confidence = clf.calculate_confidence()
    assert distance_confidence.shape == (3, len(np.unique(y)))


# Test basic functionality of EnsembleDistanceClassifier
def test_ensemble_distance_classifier():
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        shuffle=True,
    )
    clf = EnsembleDistanceClassifier(feat_idx=0)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))
