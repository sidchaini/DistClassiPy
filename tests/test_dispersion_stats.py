"""Tests for the dispersion_stat options."""

import numpy as np

import pytest

from sklearn.datasets import make_classification

import distclassipy as dcpy
from distclassipy.anomaly import DistanceAnomaly

STATS = ["std", "iqr"]


@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_classes=3, random_state=7
    )
    return X - X.min() + 0.1, y  # shift positive: many metrics assume >= 0


@pytest.mark.parametrize("stat", STATS)
def test_fit_predict_runs(data, stat):
    X, y = data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat=stat)
    clf.fit(X, y)
    pred = clf.predict(X, metric="canberra")
    assert pred.shape == (X.shape[0],)
    assert set(np.unique(pred)) <= set(np.unique(y))


def test_std_path_unchanged(data):
    """The refactor must not perturb the original std scaling at all."""
    X, y = data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="std").fit(X, y)
    XA, XB = clf._class_transform(X, clf.classes_[0])
    w = 1 / np.clip(
        clf.df_std_.loc[clf.classes_[0]].to_numpy(), np.finfo(float).eps, None
    )
    np.testing.assert_array_equal(XA, X * w)
    np.testing.assert_array_equal(
        XB, clf.df_centroid_.loc[clf.classes_[0]].to_numpy().reshape(1, -1) * w
    )


@pytest.mark.parametrize("stat", STATS)
def test_anomaly_detector_accepts_stat(data, stat):
    X, y = data
    det = DistanceAnomaly(dispersion_stat=stat).fit(X, y)
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("stat", STATS)
def test_sklearn_clone_roundtrip(stat):
    from sklearn.base import clone

    clf = dcpy.DistanceMetricClassifier(dispersion_stat=stat)
    c2 = clone(clf)
    assert c2.get_params()["dispersion_stat"] == stat


def test_unknown_stat_raises(data):
    X, y = data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="nonsense").fit(X, y)
    with pytest.raises(ValueError, match="Unknown dispersion_stat"):
        clf.predict(X, metric="euclidean")
