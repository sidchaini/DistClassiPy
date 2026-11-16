"""Tests for EnsembleDistanceClassifier and its feature-selection helpers."""

import numpy as np

import pandas as pd

import pytest

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score

from distclassipy import DistanceMetricClassifier
from distclassipy.ensemble import _forward_sfs, _one_sigma_subset


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
