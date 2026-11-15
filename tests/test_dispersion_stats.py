"""Tests for the aiqr / cdf dispersion options (v0.4.0)."""

import numpy as np

import pytest

from sklearn.datasets import make_classification

import distclassipy as dcpy
from distclassipy.anomaly import DistanceAnomaly

STATS = ["std", "iqr", "aiqr", "astd", "cdf"]


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


def test_astd_semideviations_match_definition(skewed_data):
    """Upper/lower spreads are RMS deviations of the points on each side."""
    X, y = skewed_data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="astd").fit(X, y)
    for cl in clf.classes_:
        Xc = X[y == cl]
        c = clf.df_centroid_.loc[cl].to_numpy()
        for j in range(X.shape[1]):
            d = Xc[:, j] - c[j]
            np.testing.assert_allclose(
                clf.df_semistd_hi_.loc[cl].iloc[j], np.sqrt(np.mean(d[d > 0] ** 2))
            )
            np.testing.assert_allclose(
                clf.df_semistd_lo_.loc[cl].iloc[j], np.sqrt(np.mean(d[d < 0] ** 2))
            )


def test_astd_uses_side_specific_scale(skewed_data):
    """A point above the centroid is scaled by the upper spread, below by the lower."""
    X, y = skewed_data
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="astd").fit(X, y)
    cl = clf.classes_[0]
    c = clf.df_centroid_.loc[cl].to_numpy()
    lo = clf.df_semistd_lo_.loc[cl].to_numpy()
    hi = clf.df_semistd_hi_.loc[cl].to_numpy()
    assert np.any(hi / lo > 1.5)  # genuinely skewed fixture
    XA, XB = clf._class_transform(np.vstack([c + hi, c - lo]), cl)
    # same unit-reference convention as aiqr: one spread above -> 2, below -> 0
    np.testing.assert_allclose(XA[0], 2.0, atol=1e-6)
    np.testing.assert_allclose(XA[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(XB, 1.0, atol=1e-6)  # centroid -> reference


def test_astd_symmetric_class_has_equal_sides():
    """On symmetric data both spreads coincide (astd reduces to a std scale)."""
    z = np.random.default_rng(3).normal(size=(200, 3))
    X = np.vstack([z, -z]) + 5.0  # exactly symmetric about 5 in every feature
    y = np.zeros(len(X), dtype=int)
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="astd").fit(X, y)
    np.testing.assert_allclose(clf.df_semistd_lo_, clf.df_semistd_hi_)


def test_astd_one_sided_feature_is_floored():
    """A side with no training points gets a finite (floored) scale."""
    X = np.column_stack([np.r_[np.zeros(30), np.arange(1.0, 11.0)], np.arange(40.0)])
    y = np.zeros(len(X), dtype=int)  # feature 0: median 0, nothing below it
    clf = dcpy.DistanceMetricClassifier(dispersion_stat="astd").fit(X, y)
    XA, _ = clf._class_transform(np.array([[-1.0, 20.0]]), 0)
    assert np.all(np.isfinite(XA))


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


@pytest.mark.parametrize(
    "params, match",
    [
        (dict(dispersion_stat="nonsense"), "Unknown dispersion_stat"),
        (dict(central_stat="mode"), "Unknown central_stat"),
        (dict(dispersion_stat="cdf", cdf_grid_size=1), "cdf_grid_size"),
    ],
)
def test_invalid_params_raise_in_fit(data, params, match):
    """Bad parameters fail fast in fit(), not later in predict()."""
    X, y = data
    with pytest.raises(ValueError, match=match):
        dcpy.DistanceMetricClassifier(**params).fit(X, y)


def test_anomaly_invalid_params_raise_in_fit(data):
    X, y = data
    with pytest.raises(ValueError, match="Unknown dispersion_stat"):
        DistanceAnomaly(dispersion_stat="nonsense").fit(X, y)
