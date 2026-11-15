"""Tests for the neighbour-based class reference (reference="knn", issue #25)."""

import numpy as np

import pytest

from scipy.spatial.distance import cdist

from sklearn.base import clone

import distclassipy as dcpy
from distclassipy import classifier as classifier_module
from distclassipy.anomaly import DistanceAnomaly


@pytest.fixture
def two_class():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 3)), rng.normal(4, 1, (40, 3))]) + 10
    y = np.repeat([0, 1], [60, 40])
    return X, y


def brute_force_knn(Xq, Xc, k, metric="euclidean"):
    """Mean of the k smallest distances from each query row to Xc."""
    return np.sort(cdist(Xq, Xc, metric=metric), axis=1)[:, :k].mean(axis=1)


@pytest.mark.parametrize("k", [1, 3, 1000])
def test_knn_distance_matches_brute_force_unscaled(two_class, k):
    """Mean distance to the k nearest training objects (k capped at class size)."""
    X, y = two_class
    Xq = X[::7] + 0.3
    clf = dcpy.DistanceMetricClassifier(scale=False, reference="knn", n_neighbors=k)
    clf.fit(X, y).predict_and_analyse(Xq, metric="euclidean")
    for j, cl in enumerate(clf.classes_):
        expected = brute_force_knn(Xq, X[y == cl], min(k, (y == cl).sum()))
        np.testing.assert_allclose(clf.centroid_dist_df_.iloc[:, j], expected)


def test_knn_distance_uses_class_scaling(two_class):
    """With scale=True, query and training objects share the class's scaling."""
    X, y = two_class
    Xq = X[::5] - 0.2
    clf = dcpy.DistanceMetricClassifier(
        dispersion_stat="std", reference="knn", n_neighbors=4
    )
    clf.fit(X, y).predict_and_analyse(Xq, metric="canberra")
    for j, cl in enumerate(clf.classes_):
        w = 1 / X[y == cl].std(axis=0, ddof=1)
        expected = brute_force_knn(Xq * w, X[y == cl] * w, 4, metric="canberra")
        np.testing.assert_allclose(clf.centroid_dist_df_.iloc[:, j], expected)


def test_knn_follows_multimodal_class():
    """A two-mode class whose centroid sits on top of another class."""
    rng = np.random.default_rng(1)

    def sample(n):
        return (
            np.vstack(
                [
                    rng.normal([-6, 0], 0.5, (n, 2)),  # class 0, mode 1
                    rng.normal([6, 0], 0.5, (n, 2)),  # class 0, mode 2
                    rng.normal([0, 0], 0.5, (n, 2)),  # class 1, between them
                ]
            ),
            np.repeat([0, 0, 1], n),
        )

    X, y = sample(100)
    Xt, yt = sample(50)

    def accuracy(**params):
        clf = dcpy.DistanceMetricClassifier(scale=False, **params).fit(X, y)
        return np.mean(clf.predict(Xt) == yt)

    assert accuracy(reference="centroid") < 0.8
    assert accuracy(reference="knn", n_neighbors=5) > 0.98


def test_knn_blocking_matches_single_block(two_class, monkeypatch):
    X, y = two_class
    clf = dcpy.DistanceMetricClassifier(reference="knn", n_neighbors=3).fit(X, y)
    single = clf._distance_matrices(X, ["clark"])[0]
    monkeypatch.setattr(classifier_module, "_KNN_BLOCK_ELEMS", 7)  # many tiny blocks
    np.testing.assert_array_equal(clf._distance_matrices(X, ["clark"])[0], single)


def test_knn_exclude_self_is_leave_one_out(two_class):
    X, y = two_class
    clf = dcpy.DistanceMetricClassifier(scale=False, reference="knn", n_neighbors=1)
    clf.fit(X, y)
    with_self = clf._distance_matrices(X, ["euclidean"])[0]
    loo = clf._distance_matrices(X, ["euclidean"], exclude_self=True)[0]
    rows, own = np.arange(len(y)), np.searchsorted(clf.classes_, y)
    np.testing.assert_array_equal(with_self[rows, own], 0.0)  # finds its own copy
    for i in rows:  # ... versus the nearest *other* member of its class
        others = X[(y == y[i]) & (rows != i)]
        np.testing.assert_allclose(loo[i, own[i]], cdist(X[i : i + 1], others).min())
    other = 1 - own  # distances to the other class are unaffected
    np.testing.assert_array_equal(loo[rows, other], with_self[rows, other])


@pytest.mark.parametrize("k", [1, 5])
def test_knn_anomaly_threshold_is_calibrated_for_new_data(k):
    """Leave-one-out training scores: predict() flags ~contamination of new data.

    normalize_scores=False: per-batch min-max rescaling would mask the (roughly
    multiplicative) self-match bias for k > 1; with a single metric it is moot.
    """
    rng = np.random.default_rng(2)

    def sample(n):
        X = np.vstack([rng.normal(0, 1, (n, 3)), rng.normal(6, 1, (n, 3))]) + 10
        return X, np.repeat([0, 1], n)

    X, y = sample(750)
    Xnew, _ = sample(750)
    det = DistanceAnomaly(
        metrics=["euclidean"],
        reference="knn",
        n_neighbors=k,
        contamination=0.1,
        normalize_scores=False,
    ).fit(X, y)
    assert det.clf_.reference == "knn"  # reaches the internal classifier
    flagged = np.mean(det.predict(Xnew) == -1)
    assert 0.05 < flagged < 0.16  # self-matches would give ~1.0 (k=1), ~0.23 (k=5)


def test_knn_anomaly_ranks_outlier_highest(two_class):
    X, y = two_class
    det = DistanceAnomaly(reference="knn", n_neighbors=5).fit(X, y)
    assert det.clf_.reference == "knn"
    Xq = np.vstack([X[:20], [[40.0, 40.0, 40.0]]])
    assert det.decision_function(Xq).argmax() == len(Xq) - 1


@pytest.mark.parametrize("stat", ["std", "iqr", "aiqr", "astd", "cdf"])
def test_knn_works_with_every_dispersion_stat(two_class, stat):
    X, y = two_class
    clf = dcpy.DistanceMetricClassifier(dispersion_stat=stat, reference="knn")
    clf.fit(X, y)
    assert np.isfinite(clf._distance_matrices(X, ["canberra"])[0]).all()


@pytest.mark.parametrize(
    "params, match",
    [
        (dict(reference="density"), "Unknown reference"),
        (dict(reference="knn", n_neighbors=0), "n_neighbors"),
        (dict(reference="knn", n_neighbors=2.5), "n_neighbors"),
    ],
)
def test_knn_invalid_params_raise_in_fit(two_class, params, match):
    X, y = two_class
    with pytest.raises(ValueError, match=match):
        dcpy.DistanceMetricClassifier(**params).fit(X, y)
    with pytest.raises(ValueError, match=match):
        DistanceAnomaly(**params).fit(X, y)


def test_knn_params_survive_clone():
    for est in (
        dcpy.DistanceMetricClassifier(reference="knn", n_neighbors=7),
        DistanceAnomaly(reference="knn", n_neighbors=7),
    ):
        params = clone(est).get_params()
        assert params["reference"] == "knn" and params["n_neighbors"] == 7
