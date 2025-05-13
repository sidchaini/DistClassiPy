"""A module containing the distance metric classifier.

This module contains the DistanceMetricClassifier introduced by Chaini et al. (2024)
in "Light Curve Classification with DistClassiPy: a new distance-based classifier"


.. autoclass:: distclassipy.classifier.DistanceMetricClassifier
   :members:
   :inherited-members:
   :exclude-members: set_fit_request, set_predict_request

.. autoclass:: distclassipy.classifier.EnsembleDistanceClassifier
   :members:
   :inherited-members:
   :exclude-members: set_fit_request, set_predict_request

.. doctest-skip::

.. skip::

Copyright (C) 2024  Siddharth Chaini
-----
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

from typing import Callable, Tuple

import numpy as np

import pandas as pd

import scipy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

from . import distances
from .distances import _ALL_METRICS

# Hardcoded source packages to check for distance metrics.
METRIC_SOURCES_ = {
    "scipy.spatial.distance": scipy.spatial.distance,
    "distclassipy.distances": distances,
}


def initialize_metric_function(metric):
    """Set the metric function based on the provided metric.

    If the metric is a string, the function will look for a corresponding
    function in scipy.spatial.distance or distclassipy.distances. If the metric
    is a function, it will be used directly.
    """
    if callable(metric):
        metric_fn_ = metric
        metric_arg_ = metric

    elif isinstance(metric, str):
        metric_str_lowercase = metric.lower()
        metric_found = False
        for package_str, source in METRIC_SOURCES_.items():

            # Don't use scipy for jaccard as their implementation only works with
            # booleans - use custom jaccard instead
            if (
                package_str == "scipy.spatial.distance"
                and metric_str_lowercase == "jaccard"
            ):
                continue

            if hasattr(source, metric_str_lowercase):
                metric_fn_ = getattr(source, metric_str_lowercase)
                metric_found = True

                # Use the string as an argument if it belongs to scipy as it is
                # optimized
                metric_arg_ = (
                    metric if package_str == "scipy.spatial.distance" else metric_fn_
                )
                break
        if not metric_found:
            raise ValueError(
                f"{metric} metric not found. Please pass a string of the "
                "name of a metric in scipy.spatial.distance or "
                "distclassipy.distances, or pass a metric function directly. For a "
                "list of available metrics, see: "
                "https://sidchaini.github.io/DistClassiPy/distances.html or "
                "https://docs.scipy.org/doc/scipy/reference/spatial.distance.html"
            )
    return metric_fn_, metric_arg_


class DistanceMetricClassifier(ClassifierMixin, BaseEstimator):
    """A distance-based classifier that supports different distance metrics.

    The distance metric classifier determines the similarity between features in a
    dataset by leveraging the use of different distance metrics to. A specified
    distance metric is used to compute the distance between a given object and a
    centroid for every training class in the feature space. The classifier supports
    the use of different statistical measures for constructing the centroid and scaling
    the computed distance. Additionally, the distance metric classifier also
    optionally provides an estimate of the confidence of the classifier's predictions.

    Parameters
    ----------
    scale : bool, default=True
        Whether to scale the distance between the test object and the centroid for a
        class in the feature space. If True, the data will be scaled based on the
        specified dispersion statistic.
    central_stat : {"mean", "median"}, default="median"
        The statistic used to calculate the central tendency of the data to construct
        the feature-space centroid. Supported statistics are "mean" and "median".
    dispersion_stat : {"std", "iqr"}, default="std"
        The statistic used to calculate the dispersion of the data for scaling the
        distance. Supported  statistics are "std" for standard deviation and "iqr"
        for inter-quartile range.

        .. versionadded:: 0.1.0

    References
    ----------
    .. [1] "Light Curve Classification with DistClassiPy: a new distance-based
            classifier"

    Examples
    --------
    >>> import distclassipy as dcpy
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = dcpy.DistanceMetricClassifier()
    >>> clf.fit(X, y)
    DistanceMetricClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]], metric="canberra"))
    [0]
    """

    def __init__(
        self,
        metric: str | Callable = None,
        scale: bool = True,
        central_stat: str = "median",
        dispersion_stat: str = "std",
    ) -> None:
        """Initialize the classifier with specified parameters."""
        self.metric = metric
        self.scale = scale
        self.central_stat = central_stat
        self.dispersion_stat = dispersion_stat

    def fit(
        self, X: np.array, y: np.array, feat_labels: list[str] = None
    ) -> "DistanceMetricClassifier":
        """Calculate the feature space centroid for all classes.

        This function calculates the feature space centroid in the training
        set (X, y) for all classes using the central statistic. If scaling
        is enabled, it also calculates the appropriate dispersion statistic.
        This involves computing the centroid for every class in the feature space and
        optionally calculating the kernel density estimate and 1-dimensional distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        feat_labels : list of str, optional, default=None
            The feature labels. If not provided, default labels representing feature
            number will be used.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_data(X, y)
        self.classes_ = unique_labels(y)

        if feat_labels is None:
            feat_labels = [f"Feature_{x}" for x in range(X.shape[1])]

        centroid_list = []
        for cur_class in self.classes_:
            cur_X = X[y == cur_class]
            if self.central_stat == "median":
                centroid_list.append(np.median(cur_X, axis=0).ravel())
            elif self.central_stat == "mean":
                centroid_list.append(np.mean(cur_X, axis=0).ravel())
        df_centroid = pd.DataFrame(
            data=np.array(centroid_list), index=self.classes_, columns=feat_labels
        )
        self.df_centroid_ = df_centroid

        if self.scale and self.dispersion_stat == "std":
            std_list = []
            for cur_class in self.classes_:
                cur_X = X[y == cur_class]
                # Note we're using ddof=1 because we're dealing with a sample.
                # See more: https://stackoverflow.com/a/46083501/10743245
                std_list.append(np.std(cur_X, axis=0, ddof=1).ravel())
            df_std = pd.DataFrame(
                data=np.array(std_list), index=self.classes_, columns=feat_labels
            )
            self.df_std_ = df_std
        elif self.scale and self.dispersion_stat == "iqr":

            iqr_list = []

            for cur_class in self.classes_:
                cur_X = X[y == cur_class]
                # Note we're using ddof=1 because we're dealing with a sample.
                # See more: https://stackoverflow.com/a/46083501/10743245
                iqr_list.append(
                    np.quantile(cur_X, q=0.75, axis=0).ravel()
                    - np.quantile(cur_X, q=0.25, axis=0).ravel()
                )
            df_iqr = pd.DataFrame(
                data=np.array(iqr_list), index=self.classes_, columns=feat_labels
            )
            self.df_iqr_ = df_iqr

        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: np.array,
        metric: str | Callable = None,
    ) -> np.ndarray:
        """Predict the class labels for the provided X.

        The prediction is based on the distance of each data point in the input sample
        to the centroid for each class in the feature space. The predicted class is the
        one whose centroid is the closest to the input sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        metric : str or callable, default="euclidean"
            The distance metric to use for calculating the distance between features.

            .. versionchanged:: 0.2.0
               The metric is now specified at prediction time rather
               than during initialization, providing greater flexibility.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.

        See Also
        --------
        scipy.spatial.dist : Other distance metrics provided in SciPy
        distclassipy.distances : Distance metrics included with DistClassiPy

        Notes
        -----
        If using distance metrics supported by SciPy, it is desirable to pass a string,
        which allows SciPy to use an optimized C version of the code instead of the
        slower Python version.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        metric_to_use = metric if metric is not None else self.metric
        if metric_to_use is None:
            # defaults to euclidean
            metric_to_use = "euclidean"
        metric_fn_, metric_arg_ = initialize_metric_function(metric_to_use)

        if not self.scale:
            dist_arr = scipy.spatial.distance.cdist(
                XA=X, XB=self.df_centroid_.to_numpy(), metric=metric_arg_
            )

        else:
            dist_arr_list = []

            if self.dispersion_stat == "std":
                # Clip to avoid zero error later
                wtdf = 1 / np.clip(self.df_std_, a_min=np.finfo(float).eps, a_max=None)
            elif self.dispersion_stat == "iqr":
                wtdf = 1 / np.clip(self.df_iqr_, a_min=np.finfo(float).eps, a_max=None)

            for cl in self.classes_:
                XB = self.df_centroid_.loc[cl].to_numpy().reshape(1, -1)
                w = wtdf.loc[cl].to_numpy()  # 1/std dev
                XB = XB * w  # w is for this class only
                XA = X * w  # w is for this class only
                cl_dist = scipy.spatial.distance.cdist(XA=XA, XB=XB, metric=metric_arg_)
                dist_arr_list.append(cl_dist)
            dist_arr = np.column_stack(dist_arr_list)

        y_pred = self.classes_[dist_arr.argmin(axis=1)]
        return y_pred

    def predict_and_analyse(
        self,
        X: np.array,
        metric: str | Callable = None,
    ) -> np.ndarray:
        """Predict the class labels for the provided X and perform analysis.

        The prediction is based on the distance of each data point in the input sample
        to the centroid for each class in the feature space. The predicted class is the
        one whose centroid is the closest to the input sample.

        The analysis involves saving all calculated distances and confidences as an
        attribute for inspection and analysis later.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        metric : str or callable, default="euclidean"
            The distance metric to use for calculating the distance between features.


        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.

        See Also
        --------
        scipy.spatial.dist : Other distance metrics provided in SciPy
        distclassipy.distances : Distance metrics included with DistClassiPy

        Notes
        -----
        If using distance metrics supported by SciPy, it is desirable to pass a string,
        which allows SciPy to use an optimized C version of the code instead
        of the slower Python version.

        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        metric_to_use = metric if metric is not None else self.metric
        if metric_to_use is None:
            # defaults to euclidean
            metric_to_use = "euclidean"
        metric_fn_, metric_arg_ = initialize_metric_function(metric_to_use)

        if not self.scale:
            dist_arr = scipy.spatial.distance.cdist(
                XA=X, XB=self.df_centroid_.to_numpy(), metric=metric_arg_
            )

        else:
            dist_arr_list = []

            if self.dispersion_stat == "std":
                # Clip to avoid zero error later
                wtdf = 1 / np.clip(self.df_std_, a_min=np.finfo(float).eps, a_max=None)
            elif self.dispersion_stat == "iqr":
                wtdf = 1 / np.clip(self.df_iqr_, a_min=np.finfo(float).eps, a_max=None)

            for cl in self.classes_:
                XB = self.df_centroid_.loc[cl].to_numpy().reshape(1, -1)
                w = wtdf.loc[cl].to_numpy()  # 1/std dev
                XB = XB * w  # w is for this class only
                XA = X * w  # w is for this class only
                cl_dist = scipy.spatial.distance.cdist(XA=XA, XB=XB, metric=metric_arg_)
                dist_arr_list.append(cl_dist)
            dist_arr = np.column_stack(dist_arr_list)

        self.centroid_dist_df_ = pd.DataFrame(
            data=dist_arr, index=np.arange(X.shape[0]), columns=self.classes_
        )
        self.centroid_dist_df_.columns = [
            f"{ind}_dist" for ind in self.centroid_dist_df_.columns
        ]

        y_pred = self.classes_[dist_arr.argmin(axis=1)]

        self.analyis_ = True

        return y_pred

    def calculate_confidence(self):
        """Calculate the confidence for each prediction.

        The confidence is calculated as the inverse of the distance of each data point
        to the centroids of the training data.
        """
        check_is_fitted(self, "is_fitted_")
        if not hasattr(self, "analyis_"):
            raise ValueError(
                "Use predict_and_analyse() instead of predict() for "
                "confidence calculation."
            )

        # Calculate confidence for each prediction
        self.confidence_df_ = 1 / np.clip(
            self.centroid_dist_df_, a_min=np.finfo(float).eps, a_max=None
        )
        self.confidence_df_.columns = [
            x.replace("_dist", "_conf") for x in self.confidence_df_.columns
        ]

        return self.confidence_df_.to_numpy()

    def score(self, X, y, metric: str | Callable = None) -> float:
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        metric : str or callable, default="euclidean"
            The distance metric to use for calculating the distance between features.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        metric_to_use = metric if metric is not None else self.metric
        y_pred = self.predict(X, metric=metric_to_use)
        return accuracy_score(y, y_pred)


class EnsembleDistanceClassifier(ClassifierMixin, BaseEstimator):
    """An ensemble classifier that uses different metrics for each quantile.

    This classifier splits the data into quantiles based on a specified
    feature and uses different distance metrics for each quantile to
    construct an ensemble classifier for each quantile, generally leading
    to better performance.
    Note, however, this involves fitting the training set for each metric
    to evaluate performance, making this more computationally expensive.

    .. versionadded:: 0.2.0
    """

    def __init__(
        self,
        feat_idx: int,
        scale: bool = True,
        central_stat: str = "median",
        dispersion_stat: str = "std",
        metrics_to_consider: list[str] = None,
        random_state: int = None,
    ) -> None:
        """Initialize the classifier with specified parameters.

        Parameters
        ----------
        feat_idx : int
            The index of the feature to be used for quantile splitting.
        scale : bool, default=True
            Whether to scale the distance between the test object and the centroid.
        central_stat : str, default="median"
            The statistic used to calculate the central tendency of the data.
        dispersion_stat : str, default="std"
            The statistic used to calculate the dispersion of the data.
        metrics_to_consider : list of str, optional
            A list of distance metrics to evaluate. If None, all available
            metrics within DistClassiPy will be considered.
        random_state : int, RandomState instance or None, optional (default=None)
            Controls the randomness of the estimator. Pass an int for reproducible
            output across multiple function calls.

            .. versionadded:: 0.2.1
        """
        self.feat_idx = feat_idx
        self.scale = scale
        self.central_stat = central_stat
        self.dispersion_stat = dispersion_stat
        self.metrics_to_consider = metrics_to_consider
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, n_quantiles: int = 4
    ) -> "EnsembleDistanceClassifier":
        """Fit the ensemble classifier using the best metrics for each quantile.

        Parameters
        ----------
        X : np.ndarray
            The input feature matrix.
        y : np.ndarray
            The target labels.
        n_quantiles : int, default=4
            The number of quantiles to split the data into.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.clf_ = DistanceMetricClassifier(
            scale=self.scale,
            central_stat=self.central_stat,
            dispersion_stat=self.dispersion_stat,
        )

        # Find best metrics based on training set quantiles
        self.quantile_scores_df_, self.best_metrics_per_quantile_, self.group_bins = (
            self.evaluate_metrics(X, y, n_quantiles)
        )

        # Ensure the bins work with values outside of training data
        self.group_bins[0] = -np.inf
        self.group_bins[-1] = np.inf

        self.group_labels = [f"Quantile {i+1}" for i in range(n_quantiles)]
        self.clf_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using the best metric for each quantile.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        predictions : np.ndarray
            The predicted class labels.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        # notes for pred during best:
        # option 1:
        # loop through each metric, merge quantiles for each metric
        # pred on this
        # option 2, easier, but slower:
        # loop through each quantile, and append pred

        quantiles = pd.cut(
            X[:, self.feat_idx], bins=self.group_bins, labels=self.group_labels
        )
        grouped_data = pd.DataFrame(X).groupby(quantiles, observed=False)
        # quantile_indices = quantiles.codes  # Get integer codes for quantiles
        predictions = np.empty(X.shape[0], dtype=object)  # Change dtype to object
        for i, (lim, subdf) in enumerate(grouped_data):
            best_metric = self.best_metrics_per_quantile_.loc[self.group_labels[i]]
            preds = self.clf_.predict(subdf.to_numpy(), metric=best_metric)
            predictions[subdf.index] = preds
        # # Precompute predictions for each quantile
        # quantile_predictions = {}
        # for i, label in enumerate(self.group_labels):
        #     best_metric = self.best_metrics_per_quantile_.loc[label]
        #     quantile_data = X[quantile_indices == i]
        #     if quantile_data.size > 0:
        #         quantile_predictions[i] = self.clf_.predict(
        #             quantile_data, metric=best_metric
        #         )

        # Assign predictions to the corresponding indices
        # for i, preds in quantile_predictions.items():
        #     predictions[quantile_indices == i] = preds

        return predictions

    def evaluate_metrics(
        self, X: np.ndarray, y: np.ndarray, n_quantiles: int = 4
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Evaluate and find the best distance metrics for the specified feature.

        This method uses the standalone `find_best_metrics` function to evaluate
        different distance metrics and determine the best-performing ones for
        each quantile.

        Parameters
        ----------
        X : np.ndarray
            The input feature matrix.
        y : np.ndarray
            The target labels.
        n_quantiles : int, default=4
            The number of quantiles to split the data into.

        Returns
        -------
        quantile_scores_df : pd.DataFrame
            A DataFrame containing the accuracy scores for each metric across
            different quantiles.
        best_metrics_per_quantile : pd.Series
            A Series indicating the best-performing metric for each quantile.
        group_bins : np.ndarray
            The bins used for quantile splitting.
        """
        return find_best_metrics(
            self.clf_,
            X,
            y,
            self.feat_idx,
            n_quantiles,
            self.metrics_to_consider,
            self.random_state,
        )


def find_best_metrics(
    clf: "DistanceMetricClassifier",
    X: np.ndarray,
    y: np.ndarray,
    feat_idx: int,
    n_quantiles: int = 4,
    metrics_to_consider: list[str] = None,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Evaluate and find the best distance metrics for a given feature.

    This function evaluates different distance metrics to determine which
    performs best for a specific feature in the dataset. It splits the data
    into quantiles based on the specified feature and calculates the accuracy
    of the classifier for each metric within these quantiles.

    .. versionadded:: 0.2.0

    Parameters
    ----------
    clf : DistanceMetricClassifier
        The classifier instance to be used for evaluation.
    X : np.ndarray
        The input feature matrix.
    y : np.ndarray
        The target labels.
    feat_idx : int
        The index of the feature to be used for quantile splitting.
    n_quantiles : int, default=4
        The number of quantiles to split the data into.
    metrics_to_consider : list of str, optional
        A list of distance metrics to evaluate. If None, all available
        metrics within DistClassiPy will be considered.
    random_state : int, RandomState instance or None, optional (default=None)
        Controls the randomness of the estimator. Pass an int for reproducible
        output across multiple function calls.

        .. versionadded:: 0.2.1

    Returns
    -------
    quantile_scores_df : pd.DataFrame
        A DataFrame containing the accuracy scores for each metric across
        different quantiles.
    best_metrics_per_quantile : pd.Series
        A Series indicating the best-performing metric for each quantile.
    group_bins : np.ndarray
        The bins used for quantile splitting.
    """
    X = check_array(X)
    feature_labels = [f"Feature_{i}" for i in range(X.shape[1])]
    feature_name = f"Feature_{feat_idx}"

    if metrics_to_consider is None:
        metrics_to_consider = _ALL_METRICS

    X_df = pd.DataFrame(X, columns=feature_labels)
    y_df = pd.DataFrame(y, columns=["Target"])
    quantiles, group_bins = pd.qcut(X_df[feature_name], q=n_quantiles, retbins=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.25, stratify=quantiles, random_state=random_state
    )

    clf.fit(X_train, y_train.to_numpy().ravel())
    grouped_test_data = X_test.groupby(quantiles, observed=False)

    quantile_scores = []
    for metric in metrics_to_consider:
        scores_for_metric = [
            accuracy_score(
                y_test.loc[subdf.index], clf.predict(subdf.to_numpy(), metric=metric)
            )
            for _, subdf in grouped_test_data
        ]
        quantile_scores.append(scores_for_metric)

    quantile_scores = np.array(quantile_scores) * 100
    quantile_scores_df = pd.DataFrame(
        data=quantile_scores,
        index=metrics_to_consider,
        columns=[f"Quantile {i+1}" for i in range(n_quantiles)],
    )

    best_metrics_per_quantile = quantile_scores_df.idxmax()

    return quantile_scores_df, best_metrics_per_quantile, group_bins
