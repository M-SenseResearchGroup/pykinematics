"""
Testing of functions in pykinematics/common.py
"""
from pykinematics.common import *

import numpy as np


def test_mov_avg():
    x = np.array([1, 9, 4, 2, 1, 3, 5, 4, 3, 5, 1, 3, 8, 4, 7, 6, 1, 2])

    x_m, x_sd, pad = mov_avg(x, 5)

    assert np.allclose(x_m, np.array([0.0, 0.0, 0.0, 3.4, 3.8, 3.0, 3.0, 3.2, 4.0, 3.6, 3.2, 4.0, 4.2, 4.6, 5.6, 5.2,
                                      0.0, 0.0]))
    assert np.allclose(x_sd, np.array([0.0, 0.0, 0.0, 3.36154726, 3.1144823, 1.58113883, 1.58113883, 1.4832397, 1.0,
                                       1.67332005, 1.4832397, 2.64575131, 2.58843582, 2.88097206, 2.07364414,
                                       2.77488739, 0.0, 0.0]))
    assert np.isclose(pad, 3)


def test_find_most_still():
    np.random.seed(5)
    x1 = np.random.rand(50, 2)
    np.random.seed(34)
    x2 = np.random.rand(50, 2)
    np.random.seed(13513)
    x3 = np.random.rand(50, 2)

    still_data, idx = find_most_still((x1, x2, x3), 10, return_index=True)

    assert np.allclose(np.array(still_data), np.array([[0.49851287, 0.39499911], [0.60036786, 0.45481188],
                                                       [0.18751978, 0.54213798]]))
    assert np.isclose(idx, 7)
