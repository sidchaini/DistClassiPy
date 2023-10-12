import numpy as np
import pandas as pd
from scipy.spatial import distance
from .distances import Distance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity

class DistanceMetricClassifier(BaseEstimator, ClassifierMixin):
    """
    This class implements a distance metric classifier based on scikit-learn. The classifier uses a specified distance metric to classify data points based on their distance to a training template. The training template is created using a specified statistical measure (e.g., median or mean). The classifier can be scaled in terms of standard deviations.
    
    Parameters
    ----------
    canonical_stat : str, optional
        The statistical measure to use for creating the training template. Default is 'median'.
    metric : str or callable, optional
        The distance metric to use. Default is 'euclidean'.
    scale_std : bool, optional
        If True, classifier is scaled in terms of standard deviations. Default is True.
    """
    def __init__(self, metric: str or callable="euclidean", scale_std: bool=True,
                 canonical_stat: str="median", calculate_kde: bool=True, 
                 calculate_1d_dist: bool=True, n_jobs: int=-1):
        """
        Initialize the classifier with the given parameters.
        
        Parameters
        ----------
        metric : str or callable, optional
            The distance metric to use. Default is 'euclidean'.
        scale_std : bool, optional
            If True, classifier is scaled in terms of standard deviations. Default is True.
        canonical_stat : str, optional
            The statistical measure to use for creating the training template. Default is 'median'.
        calculate_kde : bool, optional
            If True, calculate the kernel density estimate. Default is True.
        calculate_1d_dist : bool, optional
            If True, calculate the 1-dimensional distance. Default is True.
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is -1 (use all processors).
        """
        self.metric = metric
        self.scale_std = scale_std
        self.canonical_stat = canonical_stat
        self.calculate_kde = calculate_kde
        self.calculate_1d_dist = calculate_1d_dist
        self.n_jobs = n_jobs
        self.distance_calculator = Distance()

    def fit(self, X: np.array, y: np.array, feat_labels: list[str]=None):
        """
        Fit the classifier to the data. This involves creating the training template and optionally calculating the kernel density estimate and 1-dimensional distance.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        feat_labels : list of str, optional
            The feature labels. If not provided, default labels representing feature number will be used.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        if feat_labels is None:
            feat_labels = [f"Feature_{x}" for x in range(X.shape[1])]
        
        canonical_list = []
        for cur_class in self.classes_:
            cur_X = X[np.argwhere(y == cur_class)]
            if self.canonical_stat == "median":
                canonical_list.append(np.median(cur_X, axis=0).ravel())
            elif self.canonical_stat == "mean":
                canonical_list.append(np.mean(cur_X, axis=0).ravel())
        df_canonical = pd.DataFrame(
            data=np.array(canonical_list),
            index=self.classes_,
            columns=feat_labels)
        self.df_canonical_ = df_canonical

        if self.scale_std:
            std_list = []
            for cur_class in self.classes_:
                cur_X = X[y == cur_class]
                # Note we're using ddof=1 because we're dealing with a sample. See more: https://stackoverflow.com/a/46083501/10743245
                std_list.append(np.std(cur_X, axis=0, ddof=1).ravel()) 
            df_std = pd.DataFrame(
                        data=np.array(std_list),
                        index=self.classes_,
                        columns=feat_labels)
            self.df_std_ = df_std
            
        if self.calculate_kde:
            self.set_metric_fn()
            self.kde_dict_ = {}
            
            
            for cl in self.classes_:
                subX = X[y == cl]
                # Implement the following in an if-else to save computational time.
                # kde = KernelDensity(bandwidth='scott', metric=self.metric)
                # kde.fit(subX)
                kde = KernelDensity(bandwidth='scott', metric='pyfunc', metric_params={"func": self.metric_fn_})
                kde.fit(subX)
                self.kde_dict_[cl] = kde

        self.is_fitted_ = True
        
        return self

    def predict(self, X: np.array):
        """
        Predict the class labels for the provided data. The prediction is based on the distance of each data point to the training template.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        if not self.scale_std:
            dist_arr = distance.cdist(
                    XA=X,
                    XB=self.df_canonical_.to_numpy(),
                    metric=self.metric)

        else:
            dist_arr_list = []
            wtdf = 1 / self.df_std_
            wtdf = wtdf.replace([np.inf, -np.inf], np.nan)
            wtdf = wtdf.fillna(0)

            for cl in self.classes_:
                XB = self.df_canonical_.loc[cl].to_numpy().reshape(1, -1)
                w = wtdf.loc[cl].to_numpy()  # 1/std dev
                XB = XB * w  # w is for this class only
                XA = X * w  # w is for this class only
                cl_dist = distance.cdist(XA=XA,
                                         XB=XB,
                                         metric=self.metric)
                dist_arr_list.append(cl_dist)
            dist_arr = np.column_stack(dist_arr_list)

        y_pred = self.classes_[dist_arr.argmin(axis=1)]
        return y_pred
    
    def set_metric_fn(self):
        """
        Set the metric function. If the metric is a string, the function will look for a corresponding function in scipy.spatial.distance or distances.Distance. If the metric is a function, it will be used directly.
        """
        if not callable(self.metric) or isinstance(self.metric, str):
            if hasattr(distance, self.metric):
                self.metric_fn_ = getattr(distance, self.metric)
            elif hasattr(self.distance_calculator, self.metric):    
                self.metric_fn_ = getattr(self.distance_calculator, self.metric)
            else:
                raise ValueError(f"{self.metric} metric not found. Either pass a string of the name of a metric in scipy.spatial.distance or distances.Distance, or, pass a metric function directly.")

        else:
            self.metric_fn_ = self.metric

        

    def predict_and_analyse(self, X: np.array):
        """
        Predict the class labels for the provided data and perform analysis. The analysis includes calculating the distance of each data point to the training template, and optionally calculating the kernel density estimate and 1-dimensional distance.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        if not self.scale_std:
            dist_arr = distance.cdist(
                    XA=X,
                    XB=self.df_canonical_.to_numpy(),
                    metric=self.metric)

        else:
            dist_arr_list = []
            wtdf = 1 / self.df_std_
            wtdf = wtdf.replace([np.inf, -np.inf], np.nan)
            wtdf = wtdf.fillna(0)
            self.wtdf_ = wtdf

            for cl in self.classes_:
                XB = self.df_canonical_.loc[cl].to_numpy().reshape(1, -1)
                w = wtdf.loc[cl].to_numpy()  # 1/std dev
                XB = XB * w  # w is for this class only
                XA = X * w  # w is for this class only
                cl_dist = distance.cdist(XA=XA,
                                         XB=XB,
                                         metric=self.metric)
                dist_arr_list.append(cl_dist)
            dist_arr = np.column_stack(dist_arr_list)

        self.canonical_dist_df_ = pd.DataFrame(data=dist_arr,
                                        index=np.arange(X.shape[0]),
                                        columns=self.classes_)
        self.canonical_dist_df_.columns = [f"{ind}_dist" for ind in self.canonical_dist_df_.columns]
        
        y_pred = self.classes_[dist_arr.argmin(axis=1)]
                
        if self.calculate_kde:
            # NEW: Rescale in terms of median likelihoods - calculate here
            scale_factors = np.exp([self.kde_dict_[cl].score_samples(self.df_canonical_.loc[cl].to_numpy().reshape(1, -1))[0] for cl in self.classes_])
                        
            likelihood_arr = []
            for k in self.kde_dict_.keys():
                log_pdf = self.kde_dict_[k].score_samples(X)
                likelihood_val = np.exp(log_pdf)
                likelihood_arr.append(likelihood_val)
            self.likelihood_arr_ = np.array(likelihood_arr).T
            
            # NEW: Rescale in terms of median likelihoods - rescale here
            self.likelihood_arr_ = self.likelihood_arr_ / scale_factors

        if self.calculate_1d_dist:
            conf_cl = []
            Xdf_temp = pd.DataFrame(data=X, columns=self.df_canonical_.columns)
            for cl in self.classes_:
                sum_1d_dists = np.zeros(shape=(len(Xdf_temp)))
                for feat in Xdf_temp.columns:
                    dists = distance.cdist(
                    XA=np.zeros(shape=(1, 1)),
                    XB=(self.df_canonical_.loc[cl] - Xdf_temp)[feat].to_numpy().reshape(-1, 1), 
                    metric=self.metric).ravel()
                    sum_1d_dists = sum_1d_dists + dists / self.df_std_.loc[cl, feat]
                confs = 1 / sum_1d_dists
                conf_cl.append(confs)
            conf_cl = np.array(conf_cl)
            self.conf_cl_ = conf_cl

        
        self.analyis_ = True
        
        return y_pred

    
    def calculate_confidence(self, method: str="distance_inverse"):
        """
        Calculate the confidence for each prediction. The confidence is calculated based on the distance of each data point to the training template, and optionally the kernel density estimate and 1-dimensional distance.
        
        Parameters
        ----------
        method : str, optional
            The method to use for calculating confidence. Default is 'distance_inverse'.
        """
        check_is_fitted(self, 'is_fitted_')
        if not hasattr(self, 'analyis_'):
            raise ValueError("Use predict_and_analyse() instead of predict() for confidence calculation.")
        
        # Calculate confidence for each prediction
        if method == "distance_inverse":
            self.confidence_df_ = 1 / self.canonical_dist_df_
            self.confidence_df_.columns = [x.replace("_dist", "_conf") for x in self.confidence_df_.columns]
            
        elif method == "1d_distance_inverse":
            if not self.calculate_1d_dist:
                raise ValueError("method='1d_distance_inverse' is only valid if calculate_1d_dist is set to True")
            self.confidence_df_ = pd.DataFrame(
                data=self.conf_cl_.T, 
                columns=[f"{x}_conf" for x in self.classes_])
                
        elif method == "kde_likelihood":
            if not self.calculate_kde:
                raise ValueError("method='kde_likelihood' is only valid if calculate_kde is set to True")
        
            self.confidence_df_ = pd.DataFrame(data=self.likelihood_arr_, 
                                               columns=[f"{x}_conf" for x in self.kde_dict_.keys()])
        
        return self.confidence_df_.to_numpy()