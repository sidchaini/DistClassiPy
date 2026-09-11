# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython implementations of DistClassiPy's custom distance metrics.

This module provides C-speed kernels for the distance metrics defined in
:mod:`distclassipy.distances` that have no native SciPy ``cdist`` string
support. The :func:`cdist` driver loops over all row pairs without the GIL,
matching the performance of SciPy's built-in metrics.

The pure-Python functions in :mod:`distclassipy.distances` remain the
reference implementations; every kernel here must match their numerical
semantics exactly (IEEE division behaviour under ``np.errstate(ignore)``,
``np.nansum`` nan-skipping, ``np.where`` guards and epsilon clipping).

NOTE: ``cdivision=True`` is required for correctness, not just speed: the
reference implementations rely on IEEE-754 semantics (0/0 -> nan,
x/0 -> +/-inf), which plain C double division reproduces. Without it,
Cython would raise ZeroDivisionError instead.
"""

import numpy as np

from libc.math cimport fabs, sqrt, log, pow, INFINITY
from libc.float cimport DBL_EPSILON

# Same value as distclassipy.distances.EPSILON (np.finfo(float).eps)
cdef double EPSILON = DBL_EPSILON

ctypedef double (*kernel_t)(const double* u, const double* v, Py_ssize_t n) noexcept nogil


# np.minimum/np.maximum propagate nan from either operand; libc fmin/fmax
# ignore nan, so use these instead wherever the reference uses np.minimum
# or np.maximum.
cdef inline double _min2(double a, double b) noexcept nogil:
    if a != a:
        return a
    if b != b:
        return b
    return a if a < b else b


cdef inline double _max2(double a, double b) noexcept nogil:
    if a != a:
        return a
    if b != b:
        return b
    return a if a > b else b


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

cdef double _clark(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, t
    for i in range(n):
        t = fabs(u[i] - v[i]) / (u[i] + v[i])  # IEEE: 0/0=nan, x/0=inf
        if t == t:  # nan-skip, matching np.nansum
            acc += t * t
    return sqrt(acc)


cdef double _hellinger(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, a, b, t
    for i in range(n):
        # Reference clips negatives to zero before sqrt (np.clip keeps nan)
        a = u[i] if not (u[i] < 0) else 0.0
        b = v[i] if not (v[i] < 0) else 0.0
        t = sqrt(a) - sqrt(b)
        acc += t * t
    return sqrt(2.0 * acc)


cdef double _jaccard(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double uv = 0.0, uu = 0.0, vv = 0.0
    for i in range(n):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    return 1.0 - uv / (uu + vv - uv)


cdef double _lorentzian(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0
    for i in range(n):
        acc += log(fabs(u[i] - v[i]) + 1.0)
    return acc


cdef double _marylandbridge(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double uv = 0.0, uu = 0.0, vv = 0.0
    for i in range(n):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    return 1.0 - (uv / uu + uv / vv) / 2.0


cdef double _meehl(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    # Textbook consecutive-pair formula (Deza & Deza 2013):
    # sum_{i=0}^{n-2} (u_i - v_i - u_{i+1} + v_{i+1})^2
    cdef Py_ssize_t i
    cdef double acc = 0.0, t, tt
    for i in range(n - 1):
        t = u[i] - v[i] - u[i + 1] + v[i + 1]
        tt = t * t
        if tt == tt:  # nan-skip, matching np.nansum
            acc += tt
    return acc


cdef double _motyka(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double smax = 0.0, ssum = 0.0
    for i in range(n):
        smax += _max2(u[i], v[i])
        ssum += u[i] + v[i]
    return smax / ssum


cdef double _soergel(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double sabs = 0.0, smax = 0.0
    for i in range(n):
        sabs += fabs(u[i] - v[i])
        smax += _max2(u[i], v[i])
    return sabs / smax


cdef double _wave_hedges(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, d, m
    for i in range(n):
        d = fabs(u[i] - v[i])
        m = _max2(u[i], v[i])
        if d != 0 and m != 0:  # np.where guard: 0 contribution otherwise
            acc += d / m
    return acc


cdef double _kulczynski(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double sabs = 0.0, smin = 0.0
    for i in range(n):
        sabs += fabs(u[i] - v[i])
        smin += _min2(u[i], v[i])
    return sabs / smin


cdef double _add_chisq(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, m, d
    for i in range(n):
        m = u[i] * v[i]
        if m != 0:  # np.where guard
            d = u[i] - v[i]
            acc += (d * d * (u[i] + v[i])) / m
    return acc


cdef double _acc(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    # Average of Cityblock and Chebyshev, fused in one pass.
    cdef Py_ssize_t i
    cdef double s = 0.0, mx = 0.0, t
    for i in range(n):
        t = fabs(u[i] - v[i])
        s += t
        if t != t or t > mx:  # nan-propagating max, matching np.max
            mx = t
    return (s + mx) / 2.0


cdef double _chebyshev_min(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double mn = INFINITY, t
    for i in range(n):
        t = fabs(u[i] - v[i])
        if t != t or t < mn:  # nan-propagating min, matching np.amin
            mn = t
    return mn


cdef double _czekanowski(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    # Also registered as "sorensen" (identical formula).
    cdef Py_ssize_t i
    cdef double sabs = 0.0, ssum = 0.0
    for i in range(n):
        sabs += fabs(u[i] - v[i])
        ssum += u[i] + v[i]
    return sabs / ssum


cdef double _dice(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double dd = 0.0, uu = 0.0, vv = 0.0, d
    for i in range(n):
        d = u[i] - v[i]
        dd += d * d
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    return dd / (uu + vv)


cdef double _divergence(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, d, s, t
    for i in range(n):
        d = u[i] - v[i]
        s = u[i] + v[i]
        t = (d * d) / (s * s)
        if t == t:  # nan-skip, matching np.nansum
            acc += t
    return 2.0 * acc


cdef double _google(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double x = 0.0, y = 0.0, smin = 0.0, mx, mn
    for i in range(n):
        x += u[i]
        y += v[i]
        smin += _min2(u[i], v[i])
    # Python's max([x, y]) returns y only if y > x; min likewise.
    mx = y if y > x else x
    mn = y if y < x else x
    return (mx - smin) / ((x + y) - mn)


cdef double _gower(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0
    for i in range(n):
        acc += fabs(u[i] - v[i])
    return acc / n


cdef double _jeffreys(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, up, vp, r
    for i in range(n):
        # Only exact zeros are replaced by epsilon (negatives left as-is)
        up = EPSILON if u[i] == 0 else u[i]
        vp = EPSILON if v[i] == 0 else v[i]
        r = up / vp
        if r < EPSILON:  # np.clip from below; nan fails the test and stays
            r = EPSILON
        acc += (up - vp) * log(r)
    return acc


cdef double _jensenshannon_divergence(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double dl = 0.0, dr = 0.0, up, vp, s, t1, t2
    for i in range(n):
        up = EPSILON if u[i] == 0 else u[i]
        vp = EPSILON if v[i] == 0 else v[i]
        s = up + vp
        t1 = 2.0 * up / s
        if t1 < EPSILON:
            t1 = EPSILON
        t2 = 2.0 * vp / s
        if t2 < EPSILON:
            t2 = EPSILON
        dl += up * log(t1)
        dr += vp * log(t2)
    return (dl + dr) / 2.0


cdef double _jensen_difference(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, up, vp, el1, el2
    for i in range(n):
        # np.clip(u, EPSILON, None): ALL values below eps raised (incl. negatives)
        up = EPSILON if u[i] < EPSILON else u[i]
        vp = EPSILON if v[i] < EPSILON else v[i]
        el1 = (up * log(up) + vp * log(vp)) / 2.0
        el2 = (up + vp) / 2.0
        if el2 < EPSILON:
            el2 = EPSILON
        acc += el1 - el2 * log(el2)
    return acc


cdef double _kumarjohnson(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, m, t
    for i in range(n):
        m = u[i] * v[i]
        if m != 0:  # np.where guard
            t = u[i] * u[i] - v[i] * v[i]
            # pow(negative, 1.5) -> nan, identical to np.power
            acc += (t * t) / (2.0 * pow(m, 1.5))
    return acc


cdef double _matusita(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, t
    for i in range(n):
        # No clipping (unlike hellinger): sqrt(negative) = nan propagates
        t = sqrt(u[i]) - sqrt(v[i])
        acc += t * t
    return sqrt(acc)


cdef double _penroseshape(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double umu = 0.0, vmu = 0.0, acc = 0.0, t
    for i in range(n):
        umu += u[i]
        vmu += v[i]
    umu /= n
    vmu /= n
    for i in range(n):
        t = (u[i] - umu) - (v[i] - vmu)
        acc += t * t
    return sqrt(acc)


cdef double _squared_chisq(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, s, d
    for i in range(n):
        s = u[i] + v[i]
        if s != 0:  # np.where guard
            d = u[i] - v[i]
            acc += (d * d) / s
    return acc


cdef double _prob_chisq(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    return 2.0 * _squared_chisq(u, v, n)


cdef double _ruzicka(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double smin = 0.0, smax = 0.0
    for i in range(n):
        smin += _min2(u[i], v[i])
        smax += _max2(u[i], v[i])
    return 1.0 - smin / smax


cdef double _squaredchord(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, t
    for i in range(n):
        t = sqrt(u[i]) - sqrt(v[i])
        acc += t * t
    return acc


cdef double _squared_euclidean(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, d
    for i in range(n):
        d = u[i] - v[i]
        acc += d * d
    return acc


cdef double _taneja(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, up, vp, s, la
    for i in range(n):
        up = EPSILON if u[i] == 0 else u[i]
        vp = EPSILON if v[i] == 0 else v[i]
        s = up + vp
        # sqrt(negative product) = nan; nan fails `<` and stays, like np.clip
        la = s / (2.0 * sqrt(up * vp))
        if la < EPSILON:
            la = EPSILON
        acc += (s / 2.0) * log(la)
    return acc


cdef double _tanimoto(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double su = 0.0, sv = 0.0, smin = 0.0
    for i in range(n):
        su += u[i]
        sv += v[i]
        smin += _min2(u[i], v[i])
    return (su + sv - 2.0 * smin) / (su + sv - smin)


cdef double _topsoe(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, up, vp, s, t1, t2
    for i in range(n):
        up = EPSILON if u[i] == 0 else u[i]
        vp = EPSILON if v[i] == 0 else v[i]
        s = up + vp
        t1 = 2.0 * up / s
        if t1 < EPSILON:
            t1 = EPSILON
        t2 = 2.0 * vp / s
        if t2 < EPSILON:
            t2 = EPSILON
        acc += up * log(t1) + vp * log(t2)
    return acc


cdef double _vicis_symmetric_chisq(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, d, m
    for i in range(n):
        m = _min2(u[i], v[i])
        m = m * m
        if m != 0:  # np.where guard
            d = u[i] - v[i]
            acc += (d * d) / m
    return acc


cdef double _vicis_wave_hedges(const double* u, const double* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double acc = 0.0, m
    for i in range(n):
        m = _min2(u[i], v[i])
        if m != 0:  # np.where guard
            acc += fabs(u[i] - v[i]) / m
    return acc


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

cdef kernel_t[64] _KERNELS

_METRIC_INDEX = {}


cdef void _register(str name, kernel_t f):
    cdef Py_ssize_t idx = len(_METRIC_INDEX)
    _KERNELS[idx] = f
    _METRIC_INDEX[name] = idx


_register("acc", _acc)
_register("add_chisq", _add_chisq)
_register("chebyshev_min", _chebyshev_min)
_register("clark", _clark)
_register("czekanowski", _czekanowski)
_register("dice", _dice)
_register("divergence", _divergence)
_register("google", _google)
_register("gower", _gower)
_register("hellinger", _hellinger)
_register("jaccard", _jaccard)
_register("jeffreys", _jeffreys)
_register("jensen_difference", _jensen_difference)
_register("jensenshannon_divergence", _jensenshannon_divergence)
_register("kulczynski", _kulczynski)
_register("kumarjohnson", _kumarjohnson)
_register("lorentzian", _lorentzian)
_register("marylandbridge", _marylandbridge)
_register("matusita", _matusita)
_register("meehl", _meehl)
_register("motyka", _motyka)
_register("penroseshape", _penroseshape)
_register("prob_chisq", _prob_chisq)
_register("ruzicka", _ruzicka)
_register("soergel", _soergel)
_register("sorensen", _czekanowski)  # identical formula
_register("squared_chisq", _squared_chisq)
_register("squared_euclidean", _squared_euclidean)
_register("squaredchord", _squaredchord)
_register("taneja", _taneja)
_register("tanimoto", _tanimoto)
_register("topsoe", _topsoe)
_register("vicis_symmetric_chisq", _vicis_symmetric_chisq)
_register("vicis_wave_hedges", _vicis_wave_hedges)
_register("wave_hedges", _wave_hedges)

#: Names of all metrics with a compiled kernel (public API for dispatch).
CYTHON_METRICS = frozenset(_METRIC_INDEX)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def cdist(XA, XB, str metric):
    """Compute the pairwise distance matrix between two collections.

    A C-speed, reduced-scope drop-in for ``scipy.spatial.distance.cdist``
    supporting the custom metrics implemented in this module.

    Parameters
    ----------
    XA : array-like of shape (mA, n)
    XB : array-like of shape (mB, n)
    metric : str
        Name of a metric in :data:`CYTHON_METRICS`.

    Returns
    -------
    ndarray of shape (mA, mB), dtype float64
    """
    XA = np.ascontiguousarray(XA, dtype=np.float64)
    XB = np.ascontiguousarray(XB, dtype=np.float64)
    if XA.ndim != 2 or XB.ndim != 2:
        raise ValueError("XA and XB must be 2-dimensional.")
    if XA.shape[1] != XB.shape[1]:
        raise ValueError("XA and XB must have the same number of columns.")

    cdef Py_ssize_t idx
    try:
        idx = _METRIC_INDEX[metric.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown metric {metric!r}; supported metrics: "
            f"{sorted(CYTHON_METRICS)}"
        ) from None

    cdef double[:, ::1] A = XA
    cdef double[:, ::1] B = XB
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
    cdef double[:, ::1] D = out
    cdef kernel_t f = _KERNELS[idx]
    cdef Py_ssize_t i, j, n = A.shape[1]
    with nogil:
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                D[i, j] = f(&A[i, 0], &B[j, 0], n)
    return out
