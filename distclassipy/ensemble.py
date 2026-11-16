"""An ensemble of distance metric classifiers, each using its own features.

Copyright (C) 2024  Siddharth Chaini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numbers

import numpy as np

import pandas as pd

from joblib import Parallel, delayed

from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from .classifier import DistanceMetricClassifier, initialize_metric_function
from .distances import _UNIQUE_METRICS

_WEIGHTS = ("f1", "uniform")


def _forward_sfs(estimator, X, y, cv, max_features, n_jobs) -> pd.DataFrame:
    """Forward sequential feature selection, keeping the best subset of each size.

    At each step, every remaining feature is added in turn to the current
    subset and scored by the mean macro F1 over unshuffled stratified ``cv``
    folds; the best addition (first one on ties) is kept. This follows the
    selection used in the DistClassiPy paper (mlxtend's forward SFS).

    Returns a DataFrame indexed by subset size (``num_feats``) with the
    columns ``feature_idx`` (ascending tuple of column indices),
    ``avg_score``, ``std_dev`` (``np.std`` of the fold scores) and
    ``cv_scores``.
    """
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    max_features = min(max_features, n_features)
    folds = StratifiedKFold(cv)

    selected = []
    rows = []
    for _ in range(max_features):
        candidates = [j for j in range(n_features) if j not in selected]
        fold_scores = Parallel(n_jobs=n_jobs)(
            delayed(cross_val_score)(
                clone(estimator),
                X[:, sorted(selected + [j])],
                y,
                cv=folds,
                scoring="f1_macro",
            )
            for j in candidates
        )
        best = int(np.argmax([scores.mean() for scores in fold_scores]))
        selected = sorted(selected + [candidates[best]])
        rows.append(
            dict(
                feature_idx=tuple(selected),
                avg_score=fold_scores[best].mean(),
                std_dev=np.std(fold_scores[best]),
                cv_scores=fold_scores[best],
            )
        )
    return pd.DataFrame(
        rows, index=pd.Index(range(1, max_features + 1), name="num_feats")
    )


def _one_sigma_subset(path: pd.DataFrame) -> list:
    """Smallest subset whose mean score is within one std of the best mean."""
    best = path["avg_score"].idxmax()
    threshold = path.loc[best, "avg_score"] - path.loc[best, "std_dev"]
    within = path[path["avg_score"] >= threshold].sort_index()
    return list(within.iloc[0]["feature_idx"])


def _member_distances(member: DistanceMetricClassifier, X: np.ndarray) -> np.ndarray:
    """Class distances (n_samples, n_classes) of a fitted member, as in predict."""
    metric_arg = initialize_metric_function(member.metric)[1]
    return member._distance_matrices(X, [metric_arg])[0]


def _log_proba(D: np.ndarray, tau: float) -> np.ndarray:
    """Log class probabilities ``log softmax(-D / tau)`` from class distances.

    Undefined (NaN or infinite) distances give probability 0; an object with
    no finite distance gets uniform probabilities.
    """
    logits = np.where(np.isfinite(D), -D / tau, -np.inf)
    logits[~np.isfinite(logits).any(axis=1)] = 0.0
    return log_softmax(logits, axis=1)


def _fit_temperature(D: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature minimising the log loss of ``softmax(-D / tau)``.

    The search runs over ``log(tau / scale)`` in [-10, 10], where ``scale`` is
    the median spread of each object's finite class distances, so that it
    adapts to the very different distance scales of different metrics.
    """
    finite = np.where(np.isfinite(D), D, np.nan)
    spread = np.nanmax(finite, axis=1) - np.nanmin(finite, axis=1)
    scale = np.nanmedian(spread) if np.isfinite(spread).any() else 1.0
    if not scale > 0:
        scale = 1.0
    rows = np.arange(len(y_idx))

    def log_loss(log_tau):
        log_p = _log_proba(D, scale * np.exp(log_tau))[rows, y_idx]
        return -np.maximum(log_p, -50.0).mean()  # bounded loss for p = 0

    result = minimize_scalar(log_loss, bounds=(-10.0, 10.0), method="bounded")
    return float(scale * np.exp(result.x))


class EnsembleDistanceClassifier(ClassifierMixin, BaseEstimator):
    """Combine distance metric classifiers, each using its own features.

    Every metric gets its own :class:`DistanceMetricClassifier` (a *member*)
    on its own subset of the features. Each member turns its distances to the
    classes into class probabilities, ``softmax(-distance / temperature)``,
    with the temperature calibrated on internal cross-validation. The
    ensemble's probabilities are the weighted average of the members'
    probabilities, with each member weighted by its internal cross-validated
    macro F1 score. No single metric has to be chosen, and better metrics
    count more.

    Parameters
    ----------
    metrics : list of str, optional
        The metrics to combine. Defaults to ``distances._UNIQUE_METRICS``
        (16 metrics without formula-equivalent duplicates).
    feature_sets : dict, optional
        Features to use for each metric, as ``{metric: [column indices]}``,
        or ``{metric: [feature names]}`` when ``fit`` is given a DataFrame.
        Metrics without an entry get their features selected in ``fit`` by
        forward sequential feature selection, as in the DistClassiPy paper:
        macro F1 over ``cv`` folds, then the smallest subset whose score is
        within one standard deviation of the best. Entries for metrics not in
        ``metrics`` are ignored.
    scale : bool, default=True
        Passed to every member (see :class:`DistanceMetricClassifier`).
    central_stat : {"median", "mean"}, default="median"
        Passed to every member.
    dispersion_stat : {"std", "iqr", "aiqr", "astd", "cdf"}, default="std"
        Passed to every member.
    reference : {"knn", "centroid"}, default="knn"
        Passed to every member: the distance to a class is the mean distance
        to its ``n_neighbors`` nearest training objects ("knn"), or the
        distance to its centroid.
    n_neighbors : int, default=15
        Passed to every member.
    weights : {"f1", "uniform"}, default="f1"
        Member weights: proportional to each member's internal
        cross-validated macro F1, or equal.
    cv : int, default=5
        Number of unshuffled stratified folds used for feature selection,
        member scores and temperature calibration.
    max_features : int, optional
        Largest feature subset tried by the feature selection (default: all
        features). Lower values make ``fit`` faster.
    n_jobs : int, optional
        Number of jobs for the feature selection (joblib convention).

    Attributes
    ----------
    classes_ : ndarray
        The class labels.
    metrics_ : list of str
        The metrics combined.
    estimators_ : dict
        Fitted member for each metric, trained on that metric's features.
    feature_sets_ : dict
        Column indices used by each metric's member.
    feature_selection_ : dict
        Feature-selection path (best subset and score for each subset size)
        for each metric whose features were selected in ``fit``.
    cv_scores_ : pandas.Series
        Out-of-fold macro F1 of each member.
    temperatures_ : pandas.Series
        Calibrated temperature of each member.
    weights_ : pandas.Series
        Weight of each member (sums to 1).

    Examples
    --------
    >>> import distclassipy as dcpy
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    >>> clf = dcpy.EnsembleDistanceClassifier(metrics=["canberra", "clark"])
    >>> clf.fit(X, y).predict(X[:3])  # doctest: +SKIP
    """

    def __init__(
        self,
        metrics: list = None,
        feature_sets: dict = None,
        scale: bool = True,
        central_stat: str = "median",
        dispersion_stat: str = "std",
        reference: str = "knn",
        n_neighbors: int = 15,
        weights: str = "f1",
        cv: int = 5,
        max_features: int = None,
        n_jobs: int = None,
    ) -> None:
        """Initialize the ensemble with the specified parameters."""
        self.metrics = metrics
        self.feature_sets = feature_sets
        self.scale = scale
        self.central_stat = central_stat
        self.dispersion_stat = dispersion_stat
        self.reference = reference
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.cv = cv
        self.max_features = max_features
        self.n_jobs = n_jobs

    def _check_params(self) -> None:
        """Raise ValueError for invalid ensemble-level parameters."""
        if self.weights not in _WEIGHTS:
            raise ValueError(
                f"Unknown weights: {self.weights!r}; expected one of {_WEIGHTS}."
            )
        if not (isinstance(self.cv, numbers.Integral) and self.cv >= 2):
            raise ValueError(f"cv must be an integer >= 2, got {self.cv!r}.")
        if self.max_features is not None and not (
            isinstance(self.max_features, numbers.Integral) and self.max_features >= 1
        ):
            raise ValueError(
                f"max_features must be None or an integer >= 1, "
                f"got {self.max_features!r}."
            )

    def _resolve_features(self, metric, features) -> list:
        """Column indices for a ``feature_sets`` entry (indices or names)."""
        features = list(features)
        if any(isinstance(f, str) for f in features):
            if not hasattr(self, "feature_names_in_"):
                raise ValueError(
                    f"feature_sets[{metric!r}] uses feature names, which requires "
                    "fit to be given a DataFrame with column names."
                )
            names = list(self.feature_names_in_)
            missing = [f for f in features if f not in names]
            if missing:
                raise ValueError(f"feature_sets[{metric!r}]: unknown {missing}.")
            return [names.index(f) for f in features]
        bad = [f for f in features if not 0 <= f < self.n_features_in_]
        if bad or not features:
            raise ValueError(
                f"feature_sets[{metric!r}] must be a non-empty list of column "
                f"indices in [0, {self.n_features_in_}), got {features}."
            )
        return [int(f) for f in features]

    def fit(self, X, y) -> "EnsembleDistanceClassifier":
        """Select features, calibrate and weight every member, then refit it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._check_params()
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        y_idx = np.searchsorted(self.classes_, y)
        class_counts = np.bincount(y_idx, minlength=len(self.classes_))
        if class_counts.min() < 2:
            raise ValueError(
                "Every class needs at least 2 training objects for the internal "
                f"cross-validation; class {self.classes_[class_counts.argmin()]!r} "
                f"has {class_counts.min()}."
            )
        self.metrics_ = list(_UNIQUE_METRICS if self.metrics is None else self.metrics)
        given = {} if self.feature_sets is None else self.feature_sets
        folds = StratifiedKFold(self.cv)

        self.estimators_, self.feature_sets_, self.feature_selection_ = {}, {}, {}
        cv_scores, temperatures = {}, {}
        for metric in self.metrics_:
            member = DistanceMetricClassifier(
                metric=metric,
                scale=self.scale,
                central_stat=self.central_stat,
                dispersion_stat=self.dispersion_stat,
                reference=self.reference,
                n_neighbors=self.n_neighbors,
            )
            if metric in given:
                features = self._resolve_features(metric, given[metric])
            else:
                path = _forward_sfs(
                    member, X, y, self.cv, self.max_features, self.n_jobs
                )
                self.feature_selection_[metric] = path
                features = _one_sigma_subset(path)
            X_m = X[:, features]

            # Out-of-fold class distances: member score and calibration
            D = np.empty((len(y), len(self.classes_)))
            for train, test in folds.split(X_m, y):
                fold_member = clone(member).fit(X_m[train], y[train])
                D[test] = _member_distances(fold_member, X_m[test])
            cv_scores[metric] = f1_score(
                y, self.classes_[D.argmin(axis=1)], average="macro"
            )
            temperatures[metric] = _fit_temperature(D, y_idx)

            self.estimators_[metric] = clone(member).fit(X_m, y)
            self.feature_sets_[metric] = np.array(features)

        self.cv_scores_ = pd.Series(cv_scores)
        self.temperatures_ = pd.Series(temperatures)
        weights = (
            self.cv_scores_.clip(lower=0)
            if self.weights == "f1"
            else pd.Series(1.0, index=self.cv_scores_.index)
        )
        if not weights.sum() > 0:
            weights = pd.Series(1.0, index=self.cv_scores_.index)
        self.weights_ = weights / weights.sum()
        return self

    def _member_proba(self, metric, X) -> np.ndarray:
        """Calibrated class probabilities of one member."""
        D = _member_distances(
            self.estimators_[metric], X[:, self.feature_sets_[metric]]
        )
        return np.exp(_log_proba(D, self.temperatures_[metric]))

    def predict_proba_per_metric(self, X) -> pd.DataFrame:
        """Class probabilities of every member, for inspection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        pandas.DataFrame
            Columns ``(metric, class)``; the ensemble's probabilities are the
            ``weights_``-weighted sum over metrics.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        frames = {
            metric: pd.DataFrame(self._member_proba(metric, X), columns=self.classes_)
            for metric in self.metrics_
        }
        return pd.concat(frames, axis=1, names=["metric", "class"])

    def predict_proba(self, X) -> np.ndarray:
        """Weighted average of the members' class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities, columns ordered as ``classes_``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return sum(
            self.weights_[metric] * self._member_proba(metric, X)
            for metric in self.metrics_
        )

    def predict(self, X) -> np.ndarray:
        """Predict the class with the highest ensemble probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            The predicted classes.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
