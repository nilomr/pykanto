import numpy as np
from pykanto.signal.filter import normalise


def test_normalise():
    S = np.array([[-10, -1, -20], [-10, -10, -3]])
    min_level_db = -5
    arr1 = normalise(S, min_level_db)
    arr2 = np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.4]])
    np.testing.assert_array_equal(arr1, arr2)
