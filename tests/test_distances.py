# Note: A large portion of this code has been inspired
# from https://github.com/aziele/statistical-distance/blob/f49212e/test.py

from distclassipy import distances

import numpy as np

import pytest

# Test case 1: non-zero values in u and v.
uv1 = [np.array([0.33, 0.21, 0.46]), np.array([0.32, 0.50, 0.18])]

# Test case 2: zero value in v.
uv2 = [np.array([0.41, 0.23, 0.36]), np.array([0.30, 0.70, 0])]

# Test case 3: zero value in u.
uv3 = [np.array([0.33, 0.67, 0]), np.array([0.50, 0.25, 0.25])]

# Test case 4: zero value in u and v.
uv4 = [np.array([0.45, 0.55, 0]), np.array([0.68, 0.32, 0])]

# Test case 5: u and v are identical.
uv5 = [
    np.array([0.20, 0.05, 0.40, 0.30, 0.05]),
    np.array([0.20, 0.05, 0.40, 0.30, 0.05]),
]


@pytest.fixture
def vectors():
    return [uv1, uv2, uv3, uv4, uv5]


def func_test(func, vectors, correct_values):
    for i, (u, v) in enumerate(vectors):
        np.testing.assert_almost_equal(func(u, v), correct_values[i], decimal=6)


def test_euclidean(vectors):
    correct_values = [0.403237, 0.602163, 0.517494, 0.325269, 0]
    func_test(distances.euclidean, vectors, correct_values)


def test_braycurtis(vectors):
    correct_values = [0.29, 0.47, 0.42, 0.23, 0]
    func_test(distances.braycurtis, vectors, correct_values)


def test_canberra(vectors):
    correct_values = [0.861335, 1.660306, 1.661341, 0.467908, 0]
    func_test(distances.canberra, vectors, correct_values)


def test_cityblock(vectors):
    correct_values = [0.58, 0.94, 0.84, 0.46, 0]
    func_test(distances.cityblock, vectors, correct_values)


def test_chebyshev(vectors):
    correct_values = [0.29, 0.47, 0.42, 0.23, 0]
    func_test(distances.chebyshev, vectors, correct_values)


def test_clark(vectors):
    correct_values = [0.598728, 1.131109, 1.118196, 0.333645, 0]
    func_test(distances.clark, vectors, correct_values)


def test_correlation(vectors):
    correct_values = [1.995478, 1.755929, 1.008617, 0.254193, 0]
    func_test(distances.correlation, vectors, correct_values)


def test_cosine(vectors):
    correct_values = [0.216689, 0.370206, 0.272996, 0.097486, 0]
    func_test(distances.cosine, vectors, correct_values)


def test_hellinger(vectors):
    correct_values = [0.5029972, 0.996069, 0.859140, 0.330477, 0]
    func_test(distances.hellinger, vectors, correct_values)


def test_jaccard(vectors):
    correct_values = [0.356579, 0.560779, 0.446110, 0.179993, 0]
    func_test(distances.jaccard, vectors, correct_values)


def test_lorentzian(vectors):
    correct_values = [0.511453, 0.797107, 0.730804, 0.414028, 0]
    func_test(distances.lorentzian, vectors, correct_values)


def test_marylandbridge(vectors):
    correct_values = [0.216404, 0.350152, 0.258621, 0.096073, 0]
    func_test(distances.marylandbridge, vectors, correct_values)


def test_meehl(vectors):
    correct_values = [0.1629, 0.3989, 0.3545, 0.2645, 0]
    func_test(distances.meehl, vectors, correct_values)


def test_motyka(vectors):
    correct_values = [0.645, 0.7349999, 0.71, 0.615, 0.5]
    func_test(distances.motyka, vectors, correct_values)


def test_soergel(vectors):
    correct_values = [0.449612, 0.639456, 0.591549, 0.373984, 0]
    func_test(distances.soergel, vectors, correct_values)


def test_wave_hedges(vectors):
    correct_values = [1.218999, 1.939721, 1.966866, 0.756417, 0]
    func_test(distances.wave_hedges, vectors, correct_values)


def test_kulczynski(vectors):
    correct_values = [0.816901, 1.773585, 1.448276, 0.597403, 0]
    func_test(distances.kulczynski, vectors, correct_values)


def test_add_chisq(vectors):
    correct_values = [1.175282, 1.345852, 1.114259, 0.456844, 0]
    func_test(distances.add_chisq, vectors, correct_values)
