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

import numbers
import warnings
from typing import Callable, Tuple

import numpy as np

import pandas as pd

import scipy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from . import distances
from ._dispatch import CYTHON_METRICS, pairwise_distance
from .distances import _ALL_METRICS

# Hardcoded source packages to check for distance metrics.
METRIC_SOURCES_ = {
    "scipy.spatial.distance": scipy.spatial.distance,
    "distclassipy.distances": distances,
}

# Accepted values for DistanceMetricClassifier parameters (checked in fit)
_CENTRAL_STATS = ("median", "mean")
_DISPERSION_STATS = ("std", "iqr", "aiqr", "astd", "cdf")
_REFERENCES = ("centroid", "knn")

# reference="knn": at most this many query-to-training distances are held in
# memory at once (query rows are processed in blocks); 2**22 float64 = 32 MiB.
_KNN_BLOCK_ELEMS = 2**22


def _mean_k_smallest(D: np.ndarray, k: int) -> np.ndarray:
    """Row-wise mean of the ``k`` smallest entries of ``D``, ignoring NaN.

    NaN entries (undefined distances, or a training object's distance to
    itself when excluded) sort last, so they are only averaged over when fewer
    than ``k`` finite values exist, and then skipped by the nan-mean.
    """
    smallest = np.partition(D, k - 1, axis=1)[:, :k]
    with warnings.catch_warnings():  # all-NaN rows -> NaN, silently
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(smallest, axis=1)


def _side_rms(d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-column root-mean-square of ``d`` over ``mask`` entries (0 if none)."""
    n = mask.sum(axis=0)
    ss = np.where(mask, d**2, 0.0).sum(axis=0)
    return np.sqrt(np.divide(ss, n, out=np.zeros_like(ss), where=n > 0))


def _asymmetric_rescale(A, centre, lo, hi):
    """Express ``A`` in side-specific spreads around a *unit* reference.

    Deviations above ``centre`` are measured in units of ``hi`` and those below
    it in units of ``lo``. The result is offset to 1 rather than 0: keeping the
    raw centre as an additive offset would make ratio-type metrics (Canberra,
    Clark, ...) evaluate |d| / (2*centre + d), which saturates to zero
    sensitivity whenever the centre is large.
    """
    return 1.0 + (A - centre) / np.where(A > centre, hi, lo)


def initialize_metric_function(metric):
    """Set the metric function based on the provided metric.

    If the metric is a string, the function will look for a corresponding
    function in scipy.spatial.distance or distclassipy.distances. If the metric
    is a function, it will be used directly.
    """
    if callable(metric):
        metric_fn_ = metric

        # If the callable is one of the distclassipy reference functions, use
        # its name so that dispatch routes to the equivalent compiled kernel.
        # The identity check ensures a user-defined function that merely
        # shares a name is not hijacked.
        name = getattr(metric, "__name__", "").lower()
        if name in CYTHON_METRICS and getattr(distances, name, None) is metric:
            metric_arg_ = name
        else:
            metric_arg_ = metric

    elif isinstance(metric, str):
        metric_str_lowercase = metric.lower()
        metric_found = False
        for package_str, source in METRIC_SOURCES_.items():

            # Don't use scipy for jaccard and dice as their implementations
            # only work with booleans - use the custom versions instead
            if package_str == "scipy.spatial.distance" and metric_str_lowercase in (
                "jaccard",
                "dice",
            ):
                continue

            if hasattr(source, metric_str_lowercase):
                metric_fn_ = getattr(source, metric_str_lowercase)
                metric_found = True

                # Use the string as an argument for scipy metrics (scipy's
                # optimized C code) and for metrics with a compiled Cython
                # kernel (routed by pairwise_distance); otherwise fall back
                # to the Python callable.
                if package_str == "scipy.spatial.distance":
                    metric_arg_ = metric
                elif metric_str_lowercase in CYTHON_METRICS:
                    metric_arg_ = metric_str_lowercase
                else:
                    metric_arg_ = metric_fn_
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
    dispersion_stat : {"std", "iqr", "aiqr", "astd", "cdf"}, default="std"
        How each class's feature space is rescaled before the distance is taken.

        - "std" : divide by the per-feature standard deviation (the original
          Mahalanobis-inspired scaling).
        - "iqr" : divide by the inter-quartile range; more robust to outliers.
        - "aiqr" : *asymmetric* quantile scaling. Deviations above the median
          are measured in units of (q3 - q2) and deviations below it in units
          of (q2 - q1), so a skewed feature is no longer forced to be
          symmetric. The map is continuous and monotone (piecewise linear with
          its kink at the median), i.e. a fixed reparametrisation of each
          feature per class, so as with "cdf" the metric axioms hold in the
          transformed space.
        - "astd" : *asymmetric* standard deviation. Like "aiqr", but the two
          sides are measured in units of the class's semi-deviations: the
          root-mean-square deviation of its training objects above, and
          below, the centroid. This keeps the second-moment sensitivity of
          "std" while allowing skew; on symmetric data both sides equal the
          standard deviation.
        - "cdf" : map each feature through the class's empirical CDF, making
          the class uniform on [0, 1]. This generalises the scaling from the
          second moment to the whole marginal distribution, and so absorbs
          skew, heavy tails and marginal multimodality at once. Because the
          map is a fixed monotone reparametrisation per class, the metric
          axioms still hold in the transformed space. Caveat: extreme values
          saturate at 0 or 1, which is desirable for classification but
          discards magnitude information for anomaly ranking.

        .. versionadded:: 0.1.0
        .. versionchanged:: 0.4.0
           Added "aiqr", "astd" and "cdf".

    cdf_grid_size : int, default=1001
        Number of quantile grid points stored per class and feature when
        ``dispersion_stat="cdf"``. Keeps memory independent of training-set
        size.

        .. versionadded:: 0.4.0

    reference : {"centroid", "knn"}, default="centroid"
        What a test object's distance to each class is measured against, in
        that class's scaled feature space.

        - "centroid" : the class centroid (see ``central_stat``). The original
          behaviour.
        - "knn" : the class's own training objects. The distance to a class is
          the mean distance to its ``n_neighbors`` nearest training objects, so
          it follows the shape of the class's distribution (several modes,
          curved or elongated classes) instead of a single prototype, and acts
          as a local density estimate. ``n_neighbors=1`` is the distance to the
          nearest training object; ``n_neighbors`` >= the class size gives the
          mean distance to all of its objects. The training data is stored, and
          prediction costs one distance per (test object, training object)
          pair.

        .. versionadded:: 0.4.0

    n_neighbors : int, default=5
        Number of nearest training objects averaged per class when
        ``reference="knn"`` (capped at the class size). Ignored otherwise.

        .. versionadded:: 0.4.0

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
        cdf_grid_size: int = 1001,
        reference: str = "centroid",
        n_neighbors: int = 5,
    ) -> None:
        """Initialize the classifier with specified parameters."""
        self.metric = metric
        self.scale = scale
        self.central_stat = central_stat
        self.dispersion_stat = dispersion_stat
        self.cdf_grid_size = cdf_grid_size
        self.reference = reference
        self.n_neighbors = n_neighbors

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
        self._validate_params()
        X, y = validate_data(self, X, y)
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

        if self.reference == "knn":
            # Each class's training objects, plus their row positions in X
            # (lets DistanceAnomaly score the training set leave-one-out).
            self.train_X_ = {c: X[y == c] for c in self.classes_}
            self._train_rows = {c: np.flatnonzero(y == c) for c in self.classes_}

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
        elif self.scale and self.dispersion_stat == "aiqr":
            # Asymmetric (two-sided) quantile scale: the half-width below the
            # median and the half-width above it are stored separately, so a
            # skewed feature is measured in the units of the side the test
            # object actually falls on.
            q1, q2, q3 = (
                np.array(
                    [np.quantile(X[y == c], q=q, axis=0).ravel() for c in self.classes_]
                )
                for q in (0.25, 0.5, 0.75)
            )
            self.df_q1_ = pd.DataFrame(q1, index=self.classes_, columns=feat_labels)
            self.df_q2_ = pd.DataFrame(q2, index=self.classes_, columns=feat_labels)
            self.df_q3_ = pd.DataFrame(q3, index=self.classes_, columns=feat_labels)
        elif self.scale and self.dispersion_stat == "astd":
            # Asymmetric standard deviation: RMS deviation of the class's
            # training objects above / below the centroid, per feature. A side
            # with no objects gets 0 here and is floored in _class_transform,
            # against the full standard deviation (also stored for that).
            lo_list, hi_list, std_list = [], [], []
            for cur_class, centre in zip(self.classes_, centroid_list):
                cur_X = X[y == cur_class]
                d = cur_X - centre
                hi_list.append(_side_rms(d, d > 0))
                lo_list.append(_side_rms(d, d < 0))
                std_list.append(np.std(cur_X, axis=0, ddof=1).ravel())
            self.df_semistd_lo_, self.df_semistd_hi_, self.df_std_ = (
                pd.DataFrame(np.array(a), index=self.classes_, columns=feat_labels)
                for a in (lo_list, hi_list, std_list)
            )
        elif self.scale and self.dispersion_stat == "cdf":
            # Full per-class marginal: each feature is mapped through that
            # class's empirical CDF, so the class becomes uniform on [0, 1]
            # regardless of skew, heavy tails or marginal multimodality.
            # A fixed quantile grid keeps the memory cost independent of the
            # training-set size.
            self.cdf_q_ = np.linspace(0.0, 1.0, self.cdf_grid_size)
            self.cdf_grid_ = {
                c: np.maximum.accumulate(
                    np.quantile(X[y == c], q=self.cdf_q_, axis=0), axis=0
                )
                for c in self.classes_
            }

        self.is_fitted_ = True

        return self

    def _validate_params(self) -> None:
        """Raise ValueError for invalid constructor parameters."""
        if self.central_stat not in _CENTRAL_STATS:
            raise ValueError(
                f"Unknown central_stat: {self.central_stat!r}; "
                f"expected one of {_CENTRAL_STATS}."
            )
        if self.dispersion_stat not in _DISPERSION_STATS:
            raise ValueError(
                f"Unknown dispersion_stat: {self.dispersion_stat!r}; "
                f"expected one of {_DISPERSION_STATS}."
            )
        if not (
            isinstance(self.cdf_grid_size, numbers.Integral) and self.cdf_grid_size >= 2
        ):
            raise ValueError(
                f"cdf_grid_size must be an integer >= 2, got {self.cdf_grid_size!r}."
            )
        if self.reference not in _REFERENCES:
            raise ValueError(
                f"Unknown reference: {self.reference!r}; expected one of {_REFERENCES}."
            )
        if not (
            isinstance(self.n_neighbors, numbers.Integral)
            and not isinstance(self.n_neighbors, bool)
            and self.n_neighbors >= 1
        ):
            raise ValueError(
                f"n_neighbors must be an integer >= 1, got {self.n_neighbors!r}."
            )

    def _class_transform(self, X: np.ndarray, cl) -> tuple:
        """Map X and class `cl`'s centroid into that class's scaled space.

        Returns ``(XA, XB)`` ready for a pairwise distance call. The ``std``
        and ``iqr`` options reproduce the original multiplicative scaling
        exactly; ``aiqr`` and ``cdf`` apply a monotone per-feature transform
        instead (see the class docstring).
        """
        XB = self.df_centroid_.loc[cl].to_numpy().reshape(1, -1)
        if not self.scale:
            return X, XB

        eps = np.finfo(float).eps
        if self.dispersion_stat in ("std", "iqr"):
            disp = self.df_std_ if self.dispersion_stat == "std" else self.df_iqr_
            w = (1 / np.clip(disp, a_min=eps, a_max=None)).loc[cl].to_numpy()
            return X * w, XB * w

        if self.dispersion_stat == "aiqr":
            q1 = self.df_q1_.loc[cl].to_numpy()
            q2 = self.df_q2_.loc[cl].to_numpy()
            q3 = self.df_q3_.loc[cl].to_numpy()
            # Floor each half-width against the full IQR, not machine epsilon:
            # spiky features routinely have one degenerate side (q2 == q1),
            # and an epsilon floor would turn that into a ~1e16 scale factor
            # that swamps every other feature.
            iqr = np.clip(q3 - q1, a_min=eps, a_max=None)
            floor = 0.05 * iqr
            lo = np.maximum(q2 - q1, floor)
            hi = np.maximum(q3 - q2, floor)
            return (
                _asymmetric_rescale(X, q2, lo, hi),
                _asymmetric_rescale(XB, q2, lo, hi),
            )

        if self.dispersion_stat == "astd":
            centre = XB.ravel()
            # Same flooring rationale as aiqr, against the full std
            std = np.clip(self.df_std_.loc[cl].to_numpy(), a_min=eps, a_max=None)
            floor = 0.05 * std
            lo = np.maximum(self.df_semistd_lo_.loc[cl].to_numpy(), floor)
            hi = np.maximum(self.df_semistd_hi_.loc[cl].to_numpy(), floor)
            return (
                _asymmetric_rescale(X, centre, lo, hi),
                _asymmetric_rescale(XB, centre, lo, hi),
            )

        if self.dispersion_stat == "cdf":
            return self._cdf_transform(X, cl), self._cdf_transform(XB, cl)

        raise ValueError(f"Unknown dispersion_stat: {self.dispersion_stat}")

    def _distance_matrices(
        self, X: np.ndarray, metric_args: list, exclude_self: bool = False
    ) -> list:
        """Distance from every row of X to every class, for each metric.

        Returns one ``(n_samples, n_classes)`` array per entry of
        ``metric_args``: the distance to each class's centroid, or with
        ``reference="knn"`` the mean distance to its ``n_neighbors`` nearest
        training objects (both in the class's scaled space). Each class's
        scaling of X is metric-independent, so classes are looped on the
        outside and each scaled copy is reused across all metrics.

        ``exclude_self=True`` means X is the training data passed to ``fit``:
        each training object's distance to itself is ignored (leave-one-out).
        It only affects ``reference="knn"``.
        """
        knn = self.reference == "knn"
        if knn and exclude_self:
            n_train = sum(len(rows) for rows in self._train_rows.values())
            if X.shape[0] != n_train:
                raise ValueError("exclude_self=True requires X to be the fit data.")
        cols = [[] for _ in metric_args]
        for cl in self.classes_:
            XA, XB = self._class_transform(X, cl)
            if knn:
                XB = self._class_transform(self.train_X_[cl], cl)[0]
                self_rows = self._train_rows[cl] if exclude_self else None
            for col, metric_arg in zip(cols, metric_args):
                if knn:
                    d = self._knn_distance(XA, XB, metric_arg, self_rows)
                    col.append(d[:, None])
                else:
                    col.append(pairwise_distance(XA, XB, metric_arg))
        return [np.column_stack(col) for col in cols]

    def _knn_distance(self, XA, XT, metric_arg, self_rows=None) -> np.ndarray:
        """Mean distance from each row of XA to its nearest rows of XT.

        ``self_rows[j]`` is the row of XA that *is* training object ``XT[j]``;
        that pair is skipped. Query rows are processed in blocks so at most
        ``_KNN_BLOCK_ELEMS`` distances are held in memory at once.
        """
        n_train = XT.shape[0]
        k = min(self.n_neighbors, n_train)
        step = max(1, _KNN_BLOCK_ELEMS // n_train)
        out = np.empty(XA.shape[0])
        for start in range(0, XA.shape[0], step):
            stop = min(start + step, XA.shape[0])
            D = np.asarray(pairwise_distance(XA[start:stop], XT, metric_arg), float)
            if self_rows is not None:
                j = np.flatnonzero((self_rows >= start) & (self_rows < stop))
                D[self_rows[j] - start, j] = np.nan
            out[start:stop] = _mean_k_smallest(D, k)
        return out

    def _cdf_transform(self, X: np.ndarray, cl) -> np.ndarray:
        """Empirical-CDF map for one class: feature values -> [0, 1] ranks."""
        grid = self.cdf_grid_[cl]
        out = np.empty(X.shape, dtype=float)
        for j in range(X.shape[1]):
            out[:, j] = np.interp(X[:, j], grid[:, j], self.cdf_q_)
        # keep strictly positive: several metrics divide by (u + v)
        return np.clip(out, np.finfo(float).eps, 1.0)

    def predict(
        self,
        X: np.array,
        metric: str | Callable = None,
    ) -> np.ndarray:
        """Predict the class labels for the provided X.

        The prediction is based on the distance of each data point in the input sample
        to the centroid for each class in the feature space (or, with
        ``reference="knn"``, to the class's nearest training objects). The predicted
        class is the closest one.

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
        Passing a metric by name (string) always selects an optimized C
        implementation: SciPy's own for SciPy-supported metrics, and
        DistClassiPy's compiled kernels for the custom metrics. Passing the
        ``distclassipy.distances`` function objects also uses the compiled
        kernels; only arbitrary user-defined callables fall back to the slower
        per-pair Python path.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        metric_to_use = metric if metric is not None else self.metric
        if metric_to_use is None:
            # defaults to euclidean
            metric_to_use = "euclidean"
        metric_fn_, metric_arg_ = initialize_metric_function(metric_to_use)

        dist_arr = self._distance_matrices(X, [metric_arg_])[0]

        y_pred = self.classes_[dist_arr.argmin(axis=1)]
        return y_pred

    def predict_and_analyse(
        self,
        X: np.array,
        metric: str | Callable = None,
    ) -> np.ndarray:
        """Predict the class labels for the provided X and perform analysis.

        The prediction is based on the distance of each data point in the input sample
        to the centroid for each class in the feature space (or, with
        ``reference="knn"``, to the class's nearest training objects). The predicted
        class is the closest one.

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
        Passing a metric by name (string) always selects an optimized C
        implementation: SciPy's own for SciPy-supported metrics, and
        DistClassiPy's compiled kernels for the custom metrics. Passing the
        ``distclassipy.distances`` function objects also uses the compiled
        kernels; only arbitrary user-defined callables fall back to the slower
        per-pair Python path.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        metric_to_use = metric if metric is not None else self.metric
        if metric_to_use is None:
            # defaults to euclidean
            metric_to_use = "euclidean"
        metric_fn_, metric_arg_ = initialize_metric_function(metric_to_use)

        dist_arr = self._distance_matrices(X, [metric_arg_])[0]

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
        X = validate_data(self, X, reset=False)

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
