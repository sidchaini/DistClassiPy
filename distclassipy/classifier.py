"""A module containing the distance metric classifier.

This module contains the DistanceMetricClassifier introduced by Chaini et al. (2024)
in "Light Curve Classification with DistClassiPy: a new distance-based classifier"

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

import warnings
from typing import Callable

import numpy as np

import pandas as pd

import scipy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .distances import Distance

# Hardcoded source packages to check for distance metrics.
METRIC_SOURCES_ = {
    "scipy.spatial.distance": scipy.spatial.distance,
    "distances.Distance": Distance(),
}


class DistanceMetricClassifier(BaseEstimator, ClassifierMixin):
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
    metric : str or callable, default="euclidean"
        The distance metric to use for calculating the distance between features.
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

    calculate_kde : bool, default=False
        Whether to calculate a kernel density estimate based confidence parameter.
        .. deprecated:: 0.2.0
            This parameter will be removed in a future version and only the
            distance confidence parameter will be available.
    calculate_1d_dist : bool, default=False
        Whether to calculate the 1-dimensional distance based confidence parameter.
        .. deprecated:: 0.2.0
            This parameter will be removed in a future version and only the
            distance confidence parameter will be available.
        Whether to calculate the 1-dimensional distance based confidence parameter.

    Attributes
    ----------
    metric : str or callable
        The distance metric used for classification.
    scale : bool
        Indicates whether the data is scaled.
    central_stat : str
        The statistic used for calculating central tendency.
    dispersion_stat : str
        The statistic used for calculating dispersion.
    calculate_kde : bool
        Indicates whether a kernel density estimate is calculated.
        .. deprecated:: 0.2.0
            This parameter will be removed in a future version.
    calculate_1d_dist : bool
        Indicates whether 1-dimensional distances are calculated.
        .. deprecated:: 0.2.0
            This parameter will be removed in a future version.

    See Also
    --------
    scipy.spatial.dist : Other distance metrics provided in SciPy
    distclassipy.Distance : Distance metrics included with DistClassiPy

    Notes
    -----
    If using distance metrics supported by SciPy, it is desirable to pass a string,
    which allows SciPy to use an optimized C version of the code instead of the slower
    Python version.

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
    >>> clf = dcpy.DistanceMetricClassifier(metric="canberra")
    >>> clf.fit(X, y)
    DistanceMetricClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [0]
    """

    def __init__(
        self,
        metric: str | Callable = "euclidean",
        scale: bool = True,
        central_stat: str = "median",
        dispersion_stat: str = "std",
        calculate_kde: bool = True,  # deprecated in 0.2.0
        calculate_1d_dist: bool = True,  # deprecated in 0.2.0
    ):
        """Initialize the classifier with specified parameters."""
        self.metric = metric
        self.scale = scale
        self.central_stat = central_stat
        self.dispersion_stat = dispersion_stat
        if calculate_kde:
            warnings.warn(
                "calculate_kde is deprecated and will be removed in version 0.2.0",
                DeprecationWarning,
            )
        self.calculate_kde = calculate_kde

        if calculate_1d_dist:
            warnings.warn(
                "calculate_1d_dist is deprecated and will be removed in version 0.2.0",
                DeprecationWarning,
            )
        self.calculate_1d_dist = calculate_1d_dist

    def initialize_metric_function(self):
        """Set the metric function based on the provided metric.

        If the metric is a string, the function will look for a corresponding
        function in scipy.spatial.distance or distances.Distance. If the metric
        is a function, it will be used directly.
        """
        if callable(self.metric):
            self.metric_fn_ = self.metric
            self.metric_arg_ = self.metric

        elif isinstance(self.metric, str):
            metric_str_lowercase = self.metric.lower()
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
                    self.metric_fn_ = getattr(source, metric_str_lowercase)
                    metric_found = True

                    # Use the string as an argument if it belongs to scipy as it is
                    # optimized
                    self.metric_arg_ = (
                        self.metric
                        if package_str == "scipy.spatial.distance"
                        else self.metric_fn_
                    )
                    break
            if not metric_found:
                raise ValueError(
                    f"{self.metric} metric not found. Please pass a string of the "
                    "name of a metric in scipy.spatial.distance or "
                    "distances.Distance, or pass a metric function directly. For a "
                    "list of available metrics, see: "
                    "https://sidchaini.github.io/DistClassiPy/distances.html or "
                    "https://docs.scipy.org/doc/scipy/reference/spatial.distance.html"
                )

    def fit(self, X: np.array, y: np.array, feat_labels: list[str] = None):
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
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[
            1
        ]  # Number of features seen during fit - required for sklearn compatibility.

        self.initialize_metric_function()

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

        if self.calculate_kde:
            warnings.warn(
                "KDE calculation is deprecated and will be removed in version 0.2.0",
                DeprecationWarning,
            )
            self.kde_dict_ = {}

            for cl in self.classes_:
                subX = X[y == cl]
                # Implement the following in an if-else to save computational time.
                # kde = KernelDensity(bandwidth='scott', metric=self.metric)
                # kde.fit(subX)
                kde = KernelDensity(
                    bandwidth="scott",
                    metric="pyfunc",
                    metric_params={"func": self.metric_fn_},
                )
                kde.fit(subX)
                self.kde_dict_[cl] = kde
        self.is_fitted_ = True

        return self

    def predict(self, X: np.array):
        """Predict the class labels for the provided X.

        The prediction is based on the distance of each data point in the input sample
        to the centroid for each class in the feature space. The predicted class is the
        one whose centroid is the closest to the input sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        if not self.scale:
            dist_arr = scipy.spatial.distance.cdist(
                XA=X, XB=self.df_centroid_.to_numpy(), metric=self.metric_arg_
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
                cl_dist = scipy.spatial.distance.cdist(
                    XA=XA, XB=XB, metric=self.metric_arg_
                )
                dist_arr_list.append(cl_dist)
            dist_arr = np.column_stack(dist_arr_list)

        y_pred = self.classes_[dist_arr.argmin(axis=1)]
        return y_pred

    def predict_and_analyse(self, X: np.array):
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

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        if not self.scale:
            dist_arr = scipy.spatial.distance.cdist(
                XA=X, XB=self.df_centroid_.to_numpy(), metric=self.metric_arg_
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
                cl_dist = scipy.spatial.distance.cdist(
                    XA=XA, XB=XB, metric=self.metric_arg_
                )
                dist_arr_list.append(cl_dist)
            dist_arr = np.column_stack(dist_arr_list)

        self.centroid_dist_df_ = pd.DataFrame(
            data=dist_arr, index=np.arange(X.shape[0]), columns=self.classes_
        )
        self.centroid_dist_df_.columns = [
            f"{ind}_dist" for ind in self.centroid_dist_df_.columns
        ]

        y_pred = self.classes_[dist_arr.argmin(axis=1)]

        if self.calculate_kde:
            warnings.warn(
                "KDE calculation in predict_and_analyse is deprecated "
                "and will be removed in version 0.2.0",
                DeprecationWarning,
            )
            # NEW: Rescale in terms of median likelihoods - calculate here
            scale_factors = np.exp(
                [
                    self.kde_dict_[cl].score_samples(
                        self.df_centroid_.loc[cl].to_numpy().reshape(1, -1)
                    )[0]
                    for cl in self.classes_
                ]
            )

            likelihood_arr = []
            for k in self.kde_dict_.keys():
                log_pdf = self.kde_dict_[k].score_samples(X)
                likelihood_val = np.exp(log_pdf)
                likelihood_arr.append(likelihood_val)
            self.likelihood_arr_ = np.array(likelihood_arr).T

            # NEW: Rescale in terms of median likelihoods - rescale here
            self.likelihood_arr_ = self.likelihood_arr_ / scale_factors
        if self.calculate_1d_dist:
            warnings.warn(
                "calculate_1d_dist is deprecated and will be removed in version 0.2.0",
                DeprecationWarning,
            )
            conf_cl = []
            Xdf_temp = pd.DataFrame(data=X, columns=self.df_centroid_.columns)
            for cl in self.classes_:
                sum_1d_dists = np.zeros(shape=(len(Xdf_temp)))
                for feat in Xdf_temp.columns:
                    dists = scipy.spatial.distance.cdist(
                        XA=np.zeros(shape=(1, 1)),
                        XB=(self.df_centroid_.loc[cl] - Xdf_temp)[feat]
                        .to_numpy()
                        .reshape(-1, 1),
                        metric=self.metric_arg_,
                    ).ravel()
                    if self.scale and self.dispersion_stat == "std":
                        sum_1d_dists = sum_1d_dists + dists / self.df_std_.loc[cl, feat]
                    elif self.scale and self.dispersion_stat == "std":
                        sum_1d_dists = sum_1d_dists + dists / self.df_iqr_.loc[cl, feat]
                    else:
                        sum_1d_dists = sum_1d_dists + dists
                confs = 1 / np.clip(sum_1d_dists, a_min=np.finfo(float).eps, a_max=None)
                conf_cl.append(confs)
            conf_cl = np.array(conf_cl)
            self.conf_cl_ = conf_cl
        self.analyis_ = True

        return y_pred

    def calculate_confidence(self, method: str = "distance_inverse"):
        """Calculate the confidence for each prediction.

        The confidence is calculated based on either the distance of each data point to
        the centroids of the training data, optionally the kernel density estimate or
        1-dimensional distance.

        Parameters
        ----------
        method : {"distance_inverse", "1d_distance_inverse", "kde_likelihood"},
                 default="distance_inverse"
            The method to use for calculating confidence. Default is
            'distance_inverse'.
            .. deprecated:: 0.2.0
                The methods '1d_distance_inverse' and
                'kde_likelihood' will be removed in version 0.2.0.
        """
        check_is_fitted(self, "is_fitted_")
        if not hasattr(self, "analyis_"):
            raise ValueError(
                "Use predict_and_analyse() instead of predict() for "
                "confidence calculation."
            )

        # Calculate confidence for each prediction
        if method == "distance_inverse":
            self.confidence_df_ = 1 / np.clip(
                self.centroid_dist_df_, a_min=np.finfo(float).eps, a_max=None
            )
            self.confidence_df_.columns = [
                x.replace("_dist", "_conf") for x in self.confidence_df_.columns
            ]

        elif method == "1d_distance_inverse":
            warnings.warn(
                "The '1d_distance_inverse' method is deprecated "
                "and will be removed in version 0.2.0",
                DeprecationWarning,
            )
            if not self.calculate_1d_dist:
                raise ValueError(
                    "method='1d_distance_inverse' is only valid if calculate_1d_dist "
                    "is set to True"
                )
            self.confidence_df_ = pd.DataFrame(
                data=self.conf_cl_.T, columns=[f"{x}_conf" for x in self.classes_]
            )

        elif method == "kde_likelihood":
            warnings.warn(
                "The 'kde_likelihood' method is deprecated and will be "
                "removed in version 0.2.0",
                DeprecationWarning,
            )
            if not self.calculate_kde:
                raise ValueError(
                    "method='kde_likelihood' is only valid if calculate_kde is set "
                    "to True"
                )

            self.confidence_df_ = pd.DataFrame(
                data=self.likelihood_arr_,
                columns=[f"{x}_conf" for x in self.kde_dict_.keys()],
            )

        return self.confidence_df_.to_numpy()
