import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import minmax_scale

from .classifier import DistanceMetricClassifier
from .distances import _UNIQUE_METRICS


class DistanceAnomaly(OutlierMixin, BaseEstimator):
    """
    An anomaly detection system based on a multi-metric distance ensemble.

    This method identifies anomalies by measuring an object's distance to the
    prototypes (centroids) of known classes. It computes this distance using
    an ensemble of different distance metrics and aggregates them to produce
    a robust anomaly score. The core hypothesis is that true anomalies will be
    distant from all known class clusters across a consensus of geometric measures.

    Parameters
    ----------
    metrics : list of str or callable, default=None
        A list of distance metrics to use for the ensemble. If None, uses a
        predefined list of 16 stable metrics from the package.

    cluster_agg : {'min', 'mean', 'median'}, default='min'
        The aggregation method for distances to different class centroids for a
        single metric.
        - 'min': An object's distance is its distance to the *nearest* known class.
        - 'mean': An object's distance is its mean distance to all *nearest* known classes.
        - 'median': A more robust measure of an object's typical distance to all classes.

    metric_agg : {'median', 'mean', 'min', 'percentile_25'}, default='median'
        The method to aggregate scores from the ensemble of metrics.
        - 'median': The median of scores across all metrics. Robust to outlier metrics.
        - 'mean': The mean of scores.
        - 'min': The minimum score across all metrics.
        - 'percentile_25': The 25th percentile of scores.

    normalize_scores : bool, default=True
        If True, applies min-max scaling to the scores from each metric before
        the final aggregation. This helps put the diverse metrics with different
        natural scales on a level playing field.

    scale : bool, default=True
        Whether to scale distances by the class dispersion. See
        DistanceMetricClassifier for details.

    central_stat : {"mean", "median"}, default="median"
        The statistic for calculating the class centroids.

    dispersion_stat : {"std", "iqr"}, default="std"
        The statistic for calculating class dispersion for scaling.

    contamination : float, default=0.1
        The proportion of outliers expected in the data set. Used for the `predict`
        method to determine the threshold for marking outliers.
    """

    def __init__(
        self,
        metrics: list = None,
        cluster_agg: str = "min",
        metric_agg: str = "median",
        normalize_scores: bool = True,
        scale: bool = True,
        central_stat: str = "median",
        dispersion_stat: str = "std",
        contamination: float = 0.1,
        **kwargs,
    ):
        self.metrics = metrics
        self.cluster_agg = cluster_agg
        self.metric_agg = metric_agg
        self.normalize_scores = normalize_scores
        self.scale = scale
        self.central_stat = central_stat
        self.dispersion_stat = dispersion_stat
        self.contamination = contamination
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DistanceAnomaly":
        """
        Fit the anomaly detector by training the underlying DistanceMetricClassifier
        on the normal data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, representing the "normal" observations.
        y : array-like of shape (n_samples,)
            The class labels for the training data.

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
        self.clf_.fit(X, y)

        if self.metrics is None:
            self.metrics_ = _UNIQUE_METRICS
        else:
            self.metrics_ = self.metrics

        # Calculate anomaly threshold based on train scores
        # train_scores = self.decision_function(X)
        # self.offset_ = np.quantile(train_scores, 1.0 - self.contamination)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the raw anomaly score for each sample. Higher scores are
        more anomalous.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score for each sample. Higher is more anomalous.
        """
        check_is_fitted(self)
        X = check_array(X)

        metric_scores = []

        for metric in self.metrics_:
            # Get dataframe for distances to all centroids from dcpy
            self.clf_.predict_and_analyse(X, metric=metric)
            dist_df = self.clf_.centroid_dist_df_

            # 1. Aggregate distances across clusters the current metric
            if self.cluster_agg == "min":
                score_for_metric = dist_df.min(axis=1).values
            elif self.cluster_agg == "median":
                score_for_metric = dist_df.median(axis=1).values
            elif self.cluster_agg == "mean":
                score_for_metric = dist_df.mean(axis=1).values
            else:
                raise ValueError(f"Unknown cluster_agg method: {self.cluster_agg}")

            metric_scores.append(score_for_metric)

        metric_scores_arr = np.array(metric_scores).T  # shape (n_samples, n_metrics)

        if self.normalize_scores:
            # Scale scores for each metric (column) to be between 0 and 1
            # Compare with Rio notebook once.
            metric_scores_arr = minmax_scale(metric_scores_arr, axis=0)

        # 2. Aggregate scores across all metrics for final anomaly score
        if self.metric_agg == "median":
            scores = np.median(metric_scores_arr, axis=1)
        elif self.metric_agg == "mean":
            scores = np.mean(metric_scores_arr, axis=1)
        elif self.metric_agg == "min":
            scores = np.min(metric_scores_arr, axis=1)
        elif self.metric_agg == "percentile_25":
            scores = np.quantile(metric_scores_arr, 0.25, axis=1)
        else:
            raise ValueError(f"Unknown metric_agg method: {self.metric_agg}")

        # # Threshold for predict() as per sklearn conventions
        # ## NOTE: DATA LEAKAGE CONCERN
        # ## FIX LATER
        # self.offset_ = np.quantile(scores, (1 - self.contamination))

        return scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the anomaly score, matching scikit-learn's convention.

        Note: Opposite of decision_function. Higher scores mean less anomalous (more normal).
        This is for compatibility with tools that expect this behavior, like IsolationForest.
        """
        return -self.decision_function(X)

    # def predict(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Predict if a particular sample is an inlier (1) or outlie (-1).

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples,)
    #         The input samples.

    #     Returns
    #     -------
    #     is_outlier : ndarray of shape (n_samples,)
    #         Returns -1 for outliers and 1 for inliers.
    #     """
    #     check_is_fitted(self)
    #     scores = self.decision_function(X)
    #     is_outlier = np.ones(X.shape[0], dtype=int)
    #     is_outlier[scores >= self.offset_] = -1
    #     return is_outlier

    # def predict(self, X: np.ndarray) -> np.ndarray:
    # NOTE: UNCOMMENT AFTER FIXING ABOVE offset_ DATA LEAKAGE CONCERN
    #     """
    #     Predict if a particular sample is an inlier or outlier.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples,)
    #         The input samples.

    #     Returns
    #     -------
    #     is_outlier : ndarray of shape (n_samples,)
    #         Returns -1 for outliers and 1 for inliers.
    #     """
    #     scores = self.decision_function(X)
    #     is_outlier = np.ones(X.shape[0], dtype=int)
    #     is_outlier[scores >= self.offset_] = -1
    #     return is_outlier


# ref:
# DOI: 10.2196/27172
