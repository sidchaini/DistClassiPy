from typing import Callable, Tuple

import numpy as np

import pandas as pd

import scipy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .distances import Distance, _ALL_METRICS


def assemble_best_classifier(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    feat_idx: int,
    n_quantiles: int = 4,
    metrics_to_consider: list = None,
) -> tuple:
    X = check_array(X)
    feature_labels = [f"Feature_{i}" for i in range(X.shape[1])]
    feature_name = f"Feature_{feat_idx}"

    if metrics_to_consider is None:
        metrics_to_consider = _ALL_METRICS

    X_df = pd.DataFrame(X, columns=feature_labels)
    y_df = pd.DataFrame(y, columns=["Target"])
    quantiles = pd.qcut(X_df[feature_name], q=n_quantiles)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.33, stratify=quantiles
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

    # todo for pred during best:
    # loop through each metric, merge quantiles for each metric
    # pred on this

    # alt, but slower:
    # loop through each quantile, and append pred

    return quantile_scores_df, best_metrics_per_quantile
