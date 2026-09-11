"""Routing layer between SciPy's cdist and the compiled custom-metric kernels.

The compiled extension is a hard requirement (like SciPy's own distance
module): importing this module fails if ``distclassipy._cdistances`` has not
been built.
"""

import scipy.spatial.distance

from ._cdistances import CYTHON_METRICS, cdist as _cy_cdist

__all__ = ["CYTHON_METRICS", "cdist", "pairwise_distance"]


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


def cdist(XA, XB, metric="euclidean"):
    """Compute the distance between each pair of the two collections.

    A drop-in convenience around :func:`scipy.spatial.distance.cdist` that
    additionally accepts all of DistClassiPy's custom metric names (e.g.
    ``"clark"``, ``"wave_hedges"``) and computes them with compiled
    C-speed kernels.

    Parameters
    ----------
    XA : array-like of shape (mA, n)
    XB : array-like of shape (mB, n)
    metric : str or callable, default="euclidean"
        Any SciPy metric name, any DistClassiPy custom metric name, or a
        callable ``f(u, v) -> float``.

    Returns
    -------
    ndarray of shape (mA, mB)

    Examples
    --------
    >>> import numpy as np
    >>> import distclassipy as dcpy
    >>> dcpy.cdist(np.eye(3), np.ones((2, 3)), metric="clark").shape
    (3, 2)
    """
    return pairwise_distance(XA, XB, metric)
