import pytest
import numpy as np
from distclassipy.classifier import DistanceMetricClassifier


# Test initialization of the classifier with specific parameters
def test_init():
    clf = DistanceMetricClassifier(metric="euclidean", scale=True)
    assert clf.metric == "euclidean"
    assert clf.scale is True


# Fix later
# # Test classifier estimator compatibility with scikit-learn
# def test_estimator_compatibility():
#     from sklearn.utils.estimator_checks import check_estimator

#     check_estimator(DistanceMetricClassifier())


# Test fitting the classifier to a dataset
def test_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier()
    clf.fit(X, y)
    assert clf.is_fitted_ is True
    assert clf.n_features_in_ == 2


# Test making predictions with the classifier
def test_predict():
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


# Test calculating confidence of predictions
def test_calculate_confidence():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(calculate_kde=True)
    clf.fit(X, y)
    clf.predict_and_analyse(X)
    confidence = clf.calculate_confidence()
    assert confidence.shape == (3, len(np.unique(y)))


# Test using different distance metrics - from scipy
def test_metric_scipy():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(metric="cityblock")
    clf.fit(X, y)
    assert clf.metric == "cityblock"


# Test using different distance metrics - from distclassipy
def test_metric_dcpy():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(metric="soergel")
    clf.fit(X, y)
    assert clf.metric == "soergel"


# Test using custom defined metric
def test_metric_custom():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values

    def metric_euc(u, v):
        return np.sqrt(np.sum((u - v) ** 2))

    clf = DistanceMetricClassifier(metric=metric_euc)
    clf.fit(X, y)
    assert callable(clf.metric)


# Test using invalid metric
def test_metric_invalid():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values

    with pytest.raises(ValueError):
        clf = DistanceMetricClassifier(metric="chaini")
        clf.fit(X, y)


# Test setting central statistical method to median
def test_central_stat_median():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(central_stat="median")
    clf.fit(X, y)
    assert clf.central_stat == "median"


# Test setting central statistical method to mean
def test_central_stat_mean():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(central_stat="mean")
    clf.fit(X, y)
    assert clf.central_stat == "mean"


# Test KDE calculation functionality
def test_kde_calculation():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(calculate_kde=True)
    clf.fit(X, y)
    clf.predict_and_analyse(X)
    assert hasattr(clf, "kde_dict_")


# Test 1D distance calculation functionality
def test_1d_distance_calculation():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(calculate_1d_dist=True)
    clf.fit(X, y)
    clf.predict_and_analyse(X)
    assert hasattr(clf, "conf_cl_")


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


# Test different methods of confidence calculation
def test_confidence_calculation_methods():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample feature set
    y = np.array([0, 1, 0])  # Sample target values
    clf = DistanceMetricClassifier(calculate_kde=True, calculate_1d_dist=True)
    clf.fit(X, y)
    clf.predict_and_analyse(X)
    distance_confidence = clf.calculate_confidence(method="distance_inverse")
    assert distance_confidence.shape == (3, len(np.unique(y)))
    kde_confidence = clf.calculate_confidence(method="kde_likelihood")
    assert kde_confidence.shape == (3, len(np.unique(y)))
    one_d_confidence = clf.calculate_confidence(method="1d_distance_inverse")
    assert one_d_confidence.shape == (3, len(np.unique(y)))
