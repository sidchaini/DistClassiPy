"""Tests for DistanceAnomaly behaviour not tied to a specific scaling option."""

import numpy as np

import pytest

from distclassipy.anomaly import DistanceAnomaly


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (150, 4)), rng.normal(5, 1, (150, 4))]) + 10
    y = np.repeat([0, 1], 150)
    X_new = np.vstack([rng.normal(0, 1, (40, 4)) + 10, rng.normal(0, 4, (20, 4)) + 30])
    return X, y, X_new


@pytest.mark.parametrize("normalize_scores", [True, False])
@pytest.mark.parametrize("reference", ["centroid", "knn"])
def test_scores_do_not_depend_on_the_batch(data, normalize_scores, reference):
    """A sample's score is the same however the input is split into batches."""
    X, y, X_new = data
    det = DistanceAnomaly(normalize_scores=normalize_scores, reference=reference)
    det.fit(X, y)
    all_at_once = det.decision_function(X_new)
    in_pieces = np.concatenate(
        [det.decision_function(X_new[:1]), det.decision_function(X_new[1:25])]
        + [det.decision_function(X_new[i : i + 1]) for i in range(25, len(X_new))]
    )
    np.testing.assert_allclose(in_pieces, all_at_once)


def test_single_outlier_scored_alone_is_flagged(data):
    """With per-batch min-max scaling, any lone sample used to score 0."""
    X, y, _ = data
    det = DistanceAnomaly().fit(X, y)
    inlier, outlier = X[:1], np.full((1, X.shape[1]), 60.0)
    assert det.decision_function(outlier)[0] > det.decision_function(inlier)[0]
    assert det.predict(outlier)[0] == -1


def test_training_scores_are_scaled_to_unit_range(data):
    """Normalisation is learned on the training scores (each metric in [0, 1])."""
    X, y, X_new = data
    det = DistanceAnomaly(metric_agg="mean").fit(X, y)
    train = det.decision_function(X)
    assert train.min() >= 0.0 and train.max() <= 1.0
    # objects beyond the training range are not squashed back into [0, 1]
    assert det.decision_function(X_new).max() > 1.0


def test_unknown_constructor_argument_is_rejected():
    """A typo must fail loudly instead of being silently ignored."""
    with pytest.raises(TypeError):
        DistanceAnomaly(n_neighbours=5)  # sic: n_neighbors
