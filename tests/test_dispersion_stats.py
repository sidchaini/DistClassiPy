"""Tests for the aiqr / cdf dispersion options (v0.4.0)."""

import numpy as np

import pytest

from sklearn.datasets import make_classification

import distclassipy as dcpy
from distclassipy.anomaly import DistanceAnomaly

STATS = ["std", "iqr", "aiqr", "cdf"]


@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_classes=3, random_state=7
    )
    return X - X.min() + 0.1, y  # shift positive: many metrics assume >= 0


@pytest.fixture
def skewed_data():
    """Strongly right-skewed features - the case aiqr/cdf are built for."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 150)
    X = np.column_stack(
        [rng.lognormal(mean=(0.0 if c else 0.6), sigma=1.2, size=300) for c in range(4)]
    )
    X[y == 1] *= 1.8
    return X, y


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


def test_cdf_maps_class_to_uniform(data):
    """Training members of a class should be ~uniform in the transformed space."""
    X, y = data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="cdf").fit(X, y)
    cl = clf.classes_[0]
    U = clf._cdf_transform(X[y == cl], cl)
    assert U.min() >= 0.0 and U.max() <= 1.0
    # a uniform sample has mean ~0.5 per feature
    np.testing.assert_allclose(U.mean(axis=0), 0.5, atol=0.06)
    # the class centroid maps to the middle
    cent = clf.df_centroid_.loc[cl].to_numpy().reshape(1, -1)
    np.testing.assert_allclose(clf._cdf_transform(cent, cl), 0.5, atol=0.02)


def test_cdf_is_monotone_and_outlier_bounded(data):
    X, y = data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="cdf").fit(X, y)
    cl = clf.classes_[0]
    probe = np.sort(np.linspace(X.min(), X.max(), 50))
    U = clf._cdf_transform(np.tile(probe[:, None], (1, X.shape[1])), cl)
    assert np.all(np.diff(U, axis=0) >= -1e-12)  # monotone non-decreasing
    wild = clf._cdf_transform(np.full((1, X.shape[1]), 1e9), cl)
    assert np.all(wild <= 1.0)  # extreme values saturate rather than explode


def test_aiqr_uses_side_specific_scale(skewed_data):
    """A point above the median is scaled by (q3-q2), below by (q2-q1)."""
    X, y = skewed_data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="aiqr").fit(X, y)
    cl = clf.classes_[0]
    q1 = clf.df_q1_.loc[cl].to_numpy()
    q2 = clf.df_q2_.loc[cl].to_numpy()
    q3 = clf.df_q3_.loc[cl].to_numpy()
    assert np.any((q3 - q2) / (q2 - q1) > 1.5)  # genuinely skewed fixture
    above, below = (q2 + (q3 - q2))[None, :], (q2 - (q2 - q1))[None, :]
    XA, XB = clf._class_transform(np.vstack([above, below]), cl)
    # results are expressed in half-widths around a unit reference, so a point
    # one half-width above the median lands at 2 and one below at 0 -- the
    # asymmetry of the raw feature is absorbed by the side-specific scale
    np.testing.assert_allclose(XA[0], 2.0, atol=1e-6)
    np.testing.assert_allclose(XA[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(XB, 1.0, atol=1e-6)  # centroid -> reference


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

    clf = dcpy.DistanceMetricClassifier(dispersion_stat=stat, cdf_grid_size=201)
    c2 = clone(clf)
    assert c2.get_params()["dispersion_stat"] == stat
    assert c2.get_params()["cdf_grid_size"] == 201


def test_unknown_stat_raises(data):
    X, y = data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="nonsense").fit(X, y)
    with pytest.raises(ValueError, match="Unknown dispersion_stat"):
        clf.predict(X, metric="euclidean")
