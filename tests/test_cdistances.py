"""Parity tests: Cython kernels vs the pure-Python reference implementations.

Every metric in ``distclassipy._cdistances.CYTHON_METRICS`` must reproduce the
output of the corresponding function in ``distclassipy.distances`` across
regular, edge-case (zeros, identical, negative) and random inputs.

Notes
-----
- The Python references are always called with *copies*: jeffreys,
  jensenshannon_divergence, taneja and topsoe mutate their inputs in place.
- Exact bitwise equality is not expected: ``np.sum`` uses pairwise summation
  while the C kernels accumulate sequentially, so results may differ in the
  last ulp. ``assert_allclose`` with a tight tolerance is the right check.
"""

from distclassipy import distances
from distclassipy._cdistances import CYTHON_METRICS, cdist as cy_cdist
from distclassipy.classifier import initialize_metric_function

from hypothesis import given, strategies as st

import numpy as np

import pytest

# The 5 edge-case pairs from test_distances.py
EDGE_PAIRS = [
    (np.array([0.33, 0.21, 0.46]), np.array([0.32, 0.50, 0.18])),
    (np.array([0.41, 0.23, 0.36]), np.array([0.30, 0.70, 0.0])),
    (np.array([0.33, 0.67, 0.0]), np.array([0.50, 0.25, 0.25])),
    (np.array([0.45, 0.55, 0.0]), np.array([0.68, 0.32, 0.0])),
    (
        np.array([0.20, 0.05, 0.40, 0.30, 0.05]),
        np.array([0.20, 0.05, 0.40, 0.30, 0.05]),
    ),
    # All-zero vectors (0/0 -> nan, x/0 -> inf paths)
    (np.zeros(4), np.zeros(4)),
    (np.array([0.1, 0.2, 0.3, 0.4]), np.zeros(4)),
    # Negative values (sqrt/log/pow nan propagation paths)
    (np.array([-0.5, 0.3, -0.1]), np.array([0.2, -0.4, 0.6])),
]

ALL_METRICS = sorted(CYTHON_METRICS)


def reference_value(name, u, v):
    """Evaluate the pure-Python reference on copies (defensive isolation)."""
    func = getattr(distances, name)
    return func(u.copy(), v.copy())


@pytest.mark.parametrize("name", ALL_METRICS)
def test_reference_does_not_mutate_inputs(name):
    # Regression guard: jeffreys/jensenshannon_divergence/taneja/topsoe used
    # to replace zeros with EPSILON in the caller's arrays
    u = np.array([0.5, 0.0, 0.3])
    v = np.array([0.0, 0.2, 0.3])
    u_orig, v_orig = u.copy(), v.copy()
    getattr(distances, name)(u, v)
    np.testing.assert_array_equal(u, u_orig)
    np.testing.assert_array_equal(v, v_orig)


@pytest.mark.parametrize("name", ALL_METRICS)
def test_pair_parity_edge_cases(name):
    for u, v in EDGE_PAIRS:
        expected = reference_value(name, u, v)
        actual = cy_cdist(u[None, :], v[None, :], name)[0, 0]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"metric={name}, u={u}, v={v}",
        )


@pytest.mark.parametrize("name", ALL_METRICS)
@pytest.mark.parametrize("n_features", [1, 2, 5, 50])
def test_matrix_parity_random(name, n_features):
    rng = np.random.default_rng(42)
    XA = rng.uniform(0.01, 10.0, size=(7, n_features))
    XB = rng.uniform(0.01, 10.0, size=(3, n_features))
    actual = cy_cdist(XA, XB, name)
    assert actual.shape == (7, 3)
    expected = np.empty((7, 3))
    for i in range(7):
        for j in range(3):
            expected[i, j] = reference_value(name, XA[i], XB[j])
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
        err_msg=f"metric={name}, n_features={n_features}",
    )


# Element strategy: mostly well-behaved values, with zeros and negatives mixed
# in to fuzz the guard/epsilon/nan code paths of the kernels
_elements = st.one_of(
    st.just(0.0),
    st.floats(
        min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=32
    ),
)

_vector_pairs = st.integers(min_value=1, max_value=20).flatmap(
    lambda n: st.tuples(
        st.lists(_elements, min_size=n, max_size=n).map(np.array),
        st.lists(_elements, min_size=n, max_size=n).map(np.array),
    )
)


@pytest.mark.parametrize("name", ALL_METRICS)
@given(_vector_pairs)
def test_property_parity_fuzz(name, data):
    """Hypothesis fuzz: compiled kernel == Python reference on arbitrary input."""
    u, v = data
    expected = reference_value(name, u, v)
    actual = cy_cdist(u[None, :], v[None, :], name)[0, 0]
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-8,
        equal_nan=True,
        err_msg=f"metric={name}, u={u!r}, v={v!r}",
    )


def test_cdist_input_validation():
    u = np.ones((2, 3))
    with pytest.raises(ValueError, match="Unknown metric"):
        cy_cdist(u, u, "not_a_metric")
    with pytest.raises(ValueError, match="2-dimensional"):
        cy_cdist(np.ones(3), u, "clark")
    with pytest.raises(ValueError, match="same number of columns"):
        cy_cdist(u, np.ones((2, 4)), "clark")


def test_cdist_accepts_non_contiguous_and_other_dtypes():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 1.0, size=(6, 8))
    sliced = X[::2, ::2]  # non-contiguous view
    expected = cy_cdist(
        np.ascontiguousarray(sliced), np.ascontiguousarray(sliced), "clark"
    )
    np.testing.assert_array_equal(cy_cdist(sliced, sliced, "clark"), expected)
    as_f32 = X.astype(np.float32)
    out = cy_cdist(as_f32, as_f32, "gower")
    assert out.dtype == np.float64


class TestDispatch:
    def test_custom_string_routes_to_cython(self):
        _, metric_arg = initialize_metric_function("clark")
        assert metric_arg == "clark"

    def test_custom_string_case_insensitive(self):
        _, metric_arg = initialize_metric_function("Clark")
        assert metric_arg == "clark"

    def test_scipy_string_stays_string(self):
        _, metric_arg = initialize_metric_function("euclidean")
        assert metric_arg == "euclidean"

    def test_dice_routes_to_distclassipy(self):
        # scipy's dice is boolean-only; the string must resolve to the
        # distclassipy implementation (same carve-out as jaccard)
        metric_fn, metric_arg = initialize_metric_function("dice")
        assert metric_fn is distances.dice
        assert metric_arg == "dice"

    def test_jaccard_routes_to_distclassipy(self):
        metric_fn, metric_arg = initialize_metric_function("jaccard")
        assert metric_fn is distances.jaccard
        assert metric_arg == "jaccard"

    def test_known_callable_routes_to_cython(self):
        _, metric_arg = initialize_metric_function(distances.clark)
        assert metric_arg == "clark"

    def test_unknown_callable_kept_as_callable(self):
        def my_metric(u, v):
            return 0.0

        _, metric_arg = initialize_metric_function(my_metric)
        assert metric_arg is my_metric

    def test_shadowing_callable_not_hijacked(self):
        # A user function that merely shares a registered name must NOT be
        # silently replaced by the Cython kernel
        def clark(u, v):
            return 42.0

        _, metric_arg = initialize_metric_function(clark)
        assert metric_arg is clark

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="metric not found"):
            initialize_metric_function("not_a_metric")


class TestEndToEnd:
    @pytest.fixture
    def data(self):
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=150,
            n_features=8,
            n_informative=5,
            n_classes=3,
            random_state=7,
        )
        # Shift positive: many metrics assume non-negative inputs
        X = X - X.min() + 0.1
        return X, y

    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_classifier_parity_fast_vs_slow_path(self, data, name, scale):
        import distclassipy as dcpy

        X, y = data
        clf = dcpy.DistanceMetricClassifier(scale=scale)
        clf.fit(X, y)

        y_fast = clf.predict(X, metric=name)

        # Force the slow scipy-callable path with a wrapper whose __name__
        # ('slow') is not a registered metric
        ref = getattr(distances, name)

        def slow(u, v, _f=ref):
            return _f(u.copy(), v.copy())

        y_slow = clf.predict(X, metric=slow)
        np.testing.assert_array_equal(y_fast, y_slow, err_msg=f"metric={name}")

    def test_classifier_parity_iqr(self, data):
        import distclassipy as dcpy

        X, y = data
        clf = dcpy.DistanceMetricClassifier(scale=True, dispersion_stat="iqr")
        clf.fit(X, y)
        for name in ["clark", "wave_hedges", "taneja"]:
            ref = getattr(distances, name)

            def slow(u, v, _f=ref):
                return _f(u.copy(), v.copy())

            np.testing.assert_array_equal(
                clf.predict(X, metric=name), clf.predict(X, metric=slow)
            )

    def test_distance_anomaly_smoke(self, data):
        from distclassipy.anomaly import DistanceAnomaly

        X, y = data
        det = DistanceAnomaly()
        det.fit(X, y)
        scores = det.decision_function(X)
        assert scores.shape == (X.shape[0],)
        assert np.all(np.isfinite(scores))
        preds = det.predict(X)
        assert set(np.unique(preds)) <= {-1, 1}

    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("cluster_agg", ["min", "median", "mean"])
    def test_distance_anomaly_matches_legacy_implementation(
        self, data, scale, cluster_agg
    ):
        # Regression guard for the optimized decision_function: replicate the
        # pre-0.3.0 implementation (predict_and_analyse per metric + pandas
        # aggregation) and require identical scores
        from sklearn.preprocessing import minmax_scale

        from distclassipy.anomaly import DistanceAnomaly

        X, y = data
        det = DistanceAnomaly(scale=scale, cluster_agg=cluster_agg)
        det.fit(X, y)
        scores_new = det.decision_function(X)

        metric_scores = []
        for metric in det.metrics_:
            det.clf_.predict_and_analyse(X, metric=metric)
            dist_df = det.clf_.centroid_dist_df_
            metric_scores.append(getattr(dist_df, cluster_agg)(axis=1).values)
        arr = np.array(metric_scores).T
        arr[arr == np.inf] = 1e9
        arr[arr == -np.inf] = -1e9
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
        arr = minmax_scale(arr, axis=0)
        scores_legacy = np.nanmedian(arr, axis=1)

        np.testing.assert_allclose(scores_new, scores_legacy, rtol=1e-12, atol=1e-12)
