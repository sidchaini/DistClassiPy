"""Benchmark: Cython custom-metric kernels vs the old Python-callable path.

Compares, for every compiled custom metric:
  - the old path: scipy.spatial.distance.cdist with a Python callable
  - the new path: distclassipy._cdistances.cdist (compiled kernel)
against scipy's native C "euclidean" as the performance-parity baseline.

Also times DistanceAnomaly.decision_function end-to-end, before (callable
wrappers forcing the slow path) vs after (default string dispatch).

Run inside the dcpy environment:
    python benchmarks/bench_cdist.py
"""

import time

import numpy as np
import scipy.spatial.distance

from distclassipy import distances
from distclassipy._cdistances import CYTHON_METRICS, cdist as cy_cdist


def best_of(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(0)
    XA = rng.uniform(0.01, 10.0, size=(10_000, 20))
    XB = rng.uniform(0.01, 10.0, size=(5, 20))

    t_scipy_euclidean = best_of(
        lambda: scipy.spatial.distance.cdist(XA, XB, metric="euclidean")
    )
    print(f"Pairwise benchmark: XA={XA.shape}, XB={XB.shape}")
    print(f"scipy 'euclidean' (C baseline): {t_scipy_euclidean * 1e3:8.2f} ms\n")

    header = (
        f"{'metric':<26}{'callable (ms)':>14}{'cython (ms)':>13}"
        f"{'speedup':>9}{'vs scipy C':>11}"
    )
    print(header)
    print("-" * len(header))

    for name in sorted(CYTHON_METRICS):
        ref = getattr(distances, name)

        def slow(u, v, _f=ref):
            return _f(u.copy(), v.copy())  # copies: some references mutate

        t_old = best_of(
            lambda: scipy.spatial.distance.cdist(XA, XB, metric=slow), repeats=2
        )
        t_new = best_of(lambda: cy_cdist(XA, XB, name))
        print(
            f"{name:<26}{t_old * 1e3:>14.1f}{t_new * 1e3:>13.2f}"
            f"{t_old / t_new:>8.0f}x{t_new / t_scipy_euclidean:>10.1f}x"
        )

    # ----- end-to-end DistanceAnomaly -----
    from distclassipy.anomaly import DistanceAnomaly
    from distclassipy.distances import _UNIQUE_METRICS

    X = rng.uniform(0.01, 10.0, size=(5_000, 10))
    y = rng.integers(0, 4, size=5_000)

    det = DistanceAnomaly()
    det.fit(X, y)

    t_fast = best_of(lambda: det.decision_function(X), repeats=3)

    def wrap(name):
        f = getattr(distances, name)

        def slow(u, v, _f=f):
            return _f(u.copy(), v.copy())

        return slow

    # Callables with unregistered names force the old scipy-callable path
    slow_metrics = [
        wrap(m) if m.lower() in CYTHON_METRICS else m for m in _UNIQUE_METRICS
    ]
    det_slow = DistanceAnomaly(metrics=slow_metrics)
    det_slow.fit(X, y)
    t_slow = best_of(lambda: det_slow.decision_function(X), repeats=1)

    print(f"\nDistanceAnomaly.decision_function on X={X.shape}, 4 classes,")
    print(f"{len(_UNIQUE_METRICS)} default metrics:")
    print(f"  old (Python callables): {t_slow:8.2f} s")
    print(f"  new (Cython dispatch):  {t_fast:8.2f} s")
    print(f"  speedup:                {t_slow / t_fast:8.0f}x")


if __name__ == "__main__":
    main()
