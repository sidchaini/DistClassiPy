"""Routing layer between SciPy's cdist and the compiled custom-metric kernels.

The compiled extension is a hard requirement (like SciPy's own distance
module): importing this module fails if ``distclassipy._cdistances`` has not
been built.
"""

import scipy.spatial.distance

from ._cdistances import CYTHON_METRICS, cdist as _cy_cdist

__all__ = ["CYTHON_METRICS", "pairwise_distance"]


def pairwise_distance(XA, XB, metric_arg):
    """Compute pairwise distances, picking the fastest available backend.

    Custom metric names (see :data:`CYTHON_METRICS`) are routed to the
    compiled Cython kernels; everything else (SciPy metric names and
    arbitrary callables) goes through ``scipy.spatial.distance.cdist``.

    Parameters
    ----------
    XA : array-like of shape (mA, n)
    XB : array-like of shape (mB, n)
    metric_arg : str or callable
        A metric name or a callable ``f(u, v) -> float``.

    Returns
    -------
    ndarray of shape (mA, mB)
    """
    if isinstance(metric_arg, str) and metric_arg.lower() in CYTHON_METRICS:
        return _cy_cdist(XA, XB, metric_arg.lower())
    return scipy.spatial.distance.cdist(XA, XB, metric=metric_arg)
