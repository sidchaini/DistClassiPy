"""An ensemble of distance metric classifiers, each using its own features.

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

import numpy as np

import pandas as pd

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


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
