"""Tests for EnsembleDistanceClassifier and its feature-selection helpers."""

import numpy as np

import pandas as pd

import pytest

from scipy.special import log_softmax

from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from distclassipy import DistanceMetricClassifier, EnsembleDistanceClassifier
from distclassipy.distances import _UNIQUE_METRICS
from distclassipy.ensemble import _forward_sfs, _one_sigma_subset

GIVEN = {"clark": [0, 1, 2], "canberra": [1, 3], "euclidean": [0, 2, 4, 5]}


@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=1,
    )
    return X - X.min() + 0.1, y  # positive: several metrics assume u, v >= 0


def member(metric):
    """A member configured like the ensemble's defaults."""
    return DistanceMetricClassifier(metric=metric, reference="knn", n_neighbors=15)


@pytest.fixture
def sfs_data():
    return make_classification(
        n_samples=300, n_features=6, n_informative=3, random_state=0
    )


def test_sfs_first_step_picks_best_single_feature(sfs_data):
    X, y = sfs_data
    clf = DistanceMetricClassifier(metric="canberra")
    path = _forward_sfs(clf, X, y, cv=5, max_features=None, n_jobs=None)
    single = [
        cross_val_score(
            clf, X[:, [j]], y, cv=StratifiedKFold(5), scoring="f1_macro"
        ).mean()
        for j in range(X.shape[1])
    ]
    assert path.loc[1, "feature_idx"] == (int(np.argmax(single)),)
    assert path.loc[1, "avg_score"] == pytest.approx(max(single))
    assert len(path.loc[1, "cv_scores"]) == 5
    assert path.loc[1, "std_dev"] == pytest.approx(np.std(path.loc[1, "cv_scores"]))


def test_sfs_path_is_nested_and_complete(sfs_data):
    X, y = sfs_data
    clf = DistanceMetricClassifier(metric="canberra")
    path = _forward_sfs(clf, X, y, cv=5, max_features=None, n_jobs=None)
    assert list(path.index) == [1, 2, 3, 4, 5, 6]
    for k in range(2, 7):
        assert set(path.loc[k - 1, "feature_idx"]) < set(path.loc[k, "feature_idx"])
        assert list(path.loc[k, "feature_idx"]) == sorted(path.loc[k, "feature_idx"])
    short = _forward_sfs(clf, X, y, cv=5, max_features=3, n_jobs=None)
    assert list(short.index) == [1, 2, 3]


def test_one_sigma_rule():
    path = pd.DataFrame(
        {
            "feature_idx": [(0,), (0, 3), (0, 1, 3), (0, 1, 2, 3)],
            "avg_score": [0.80, 0.90, 0.93, 0.92],
            "std_dev": [0.01, 0.02, 0.04, 0.03],
        },
        index=pd.Index([1, 2, 3, 4], name="num_feats"),
    )
    # 0.90 >= 0.93 - 0.04, and is the smallest such subset
    assert _one_sigma_subset(path) == [0, 3]


def test_predict_proba_is_a_distribution(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=list(GIVEN), feature_sets=GIVEN)
    proba = ens.fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(ens.predict(X), ens.classes_[proba.argmax(axis=1)])


def test_feature_sets_are_used(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=list(GIVEN), feature_sets=GIVEN)
    ens.fit(X, y)
    assert list(ens.feature_sets_["clark"]) == [0, 1, 2]
    assert ens.estimators_["clark"].n_features_in_ == 3
    assert ens.feature_selection_ == {}


def test_single_member_matches_member_prediction(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=["clark"], feature_sets=GIVEN).fit(X, y)
    alone = member("clark").fit(X[:, [0, 1, 2]], y)
    np.testing.assert_array_equal(ens.predict(X), alone.predict(X[:, [0, 1, 2]]))


def test_weights(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=list(GIVEN), feature_sets=GIVEN)
    ens.fit(X, y)
    np.testing.assert_allclose(ens.weights_, ens.cv_scores_ / ens.cv_scores_.sum())
    assert ens.weights_.sum() == pytest.approx(1.0)
    uniform = clone(ens).set_params(weights="uniform").fit(X, y)
    np.testing.assert_allclose(uniform.weights_, 1 / 3)


def test_cv_scores_are_out_of_fold_f1(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=list(GIVEN), feature_sets=GIVEN)
    ens.fit(X, y)
    oof = cross_val_predict(member("canberra"), X[:, [1, 3]], y, cv=StratifiedKFold(5))
    assert ens.cv_scores_["canberra"] == pytest.approx(
        f1_score(y, oof, average="macro")
    )


def test_temperature_minimises_oof_log_loss(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=list(GIVEN), feature_sets=GIVEN)
    ens.fit(X, y)
    X_m = X[:, [1, 3]]
    D = np.empty((len(y), 3))
    for train, test in StratifiedKFold(5).split(X_m, y):
        fold_member = member("canberra").fit(X_m[train], y[train])
        fold_member.predict_and_analyse(X_m[test])
        D[test] = fold_member.centroid_dist_df_.to_numpy()

    def oof_log_loss(tau):
        log_p = log_softmax(-D / tau, axis=1)
        return -log_p[np.arange(len(y)), np.searchsorted(ens.classes_, y)].mean()

    tau = ens.temperatures_["canberra"]
    assert tau > 0
    assert oof_log_loss(tau) <= oof_log_loss(tau / 2)
    assert oof_log_loss(tau) <= oof_log_loss(tau * 2)


def test_selects_features_when_not_given(data):
    X, y = data
    ens = EnsembleDistanceClassifier(
        metrics=["canberra", "clark"],
        feature_sets={"clark": [0, 1, 2]},
        max_features=3,
    ).fit(X, y)
    path = _forward_sfs(member("canberra"), X, y, cv=5, max_features=3, n_jobs=None)
    assert list(ens.feature_sets_["canberra"]) == _one_sigma_subset(path)
    pd.testing.assert_frame_equal(
        ens.feature_selection_["canberra"].drop(columns="cv_scores"),
        path.drop(columns="cv_scores"),
    )
    assert "clark" not in ens.feature_selection_


def test_default_metrics(data):
    X, y = data
    ens = EnsembleDistanceClassifier(
        feature_sets={m: [0, 1, 2, 3] for m in _UNIQUE_METRICS}
    ).fit(X, y)
    assert ens.metrics_ == list(_UNIQUE_METRICS)
    assert np.isfinite(ens.predict_proba(X)).all()


def test_predict_proba_per_metric(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=list(GIVEN), feature_sets=GIVEN)
    per_metric = ens.fit(X, y).predict_proba_per_metric(X[:10])
    assert per_metric.columns.names == ["metric", "class"]
    assert list(per_metric.columns.get_level_values("metric").unique()) == list(GIVEN)
    combined = sum(ens.weights_[m] * per_metric[m].to_numpy() for m in GIVEN)
    np.testing.assert_allclose(combined, ens.predict_proba(X[:10]))


def test_dataframe_with_feature_names_and_string_labels(data):
    X, y = data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_str = np.array(["a", "b", "c"])[y]
    ens = EnsembleDistanceClassifier(
        metrics=["clark", "canberra"],
        feature_sets={"clark": ["f0", "f1", "f2"], "canberra": ["f1", "f3"]},
    ).fit(X_df, y_str)
    assert list(ens.feature_sets_["clark"]) == [0, 1, 2]
    assert set(ens.predict(X_df)) <= {"a", "b", "c"}
    np.testing.assert_allclose(ens.predict_proba(X_df).sum(axis=1), 1.0)


def test_member_with_undefined_distances_votes_uniformly(data):
    X, y = data
    # correlation between one-dimensional vectors is undefined (NaN)
    ens = EnsembleDistanceClassifier(
        metrics=["correlation", "clark"],
        feature_sets={"correlation": [0], "clark": [0, 1, 2]},
    ).fit(X, y)
    proba = ens.predict_proba(X)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(ens.predict_proba_per_metric(X)["correlation"], 1 / 3)


def test_feature_names_with_an_array_raise(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=["clark"], feature_sets={"clark": ["f0"]})
    with pytest.raises(ValueError, match="feature names"):
        ens.fit(X, y)


def test_class_with_a_single_member_raises(data):
    X, y = data
    y = y.copy()
    y[0] = 3  # a class absent from the training part of one internal fold
    ens = EnsembleDistanceClassifier(metrics=["clark"], feature_sets=GIVEN)
    with pytest.raises(ValueError, match="at least 2"):
        ens.fit(X, y)


def test_class_smaller_than_cv_still_fits(data):
    X, y = data
    y = y.copy()
    y[:3] = 3  # 3 members < cv=5 folds: sklearn warns, every fold still trains on it
    ens = EnsembleDistanceClassifier(metrics=["clark"], feature_sets=GIVEN)
    with pytest.warns(UserWarning, match="least populated class"):
        ens.fit(X, y)
    assert 3 in ens.classes_


def test_sklearn_compatibility(data):
    X, y = data
    ens = EnsembleDistanceClassifier(metrics=["clark"], feature_sets=GIVEN)
    assert clone(ens).get_params() == ens.get_params()
    scores = cross_val_score(ens, X, y, cv=3)
    assert np.all((scores >= 0) & (scores <= 1))


@pytest.mark.parametrize(
    "params, match",
    [
        (dict(weights="bogus"), "weights"),
        (dict(cv=1), "cv"),
        (dict(max_features=0), "max_features"),
        (dict(feature_sets={"clark": [0, 9]}), "feature_sets"),
    ],
)
def test_invalid_params_raise_in_fit(data, params, match):
    X, y = data
    with pytest.raises(ValueError, match=match):
        EnsembleDistanceClassifier(metrics=["clark"], **params).fit(X, y)


def test_unknown_metric_raises_clearly_during_feature_selection(data):
    X, y = data
    with pytest.raises(ValueError, match="metric not found"):
        EnsembleDistanceClassifier(metrics=["clrak"], max_features=2).fit(X, y)
