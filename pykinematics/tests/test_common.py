"""
Testing of functions in pykinematics/common.py
"""
from pykinematics.common import *

import numpy as np


def test_mov_avg_window_1(simple_x):
    _, _, pad = mov_avg(simple_x, 1)

    assert pad == 1


def test_mov_avg_window_large(simple_x):
    mn, sd, pad = mov_avg(simple_x, len(simple_x) * 2 + 4)

    assert pad == 1
    assert np.allclose(mn, np.mean(simple_x))
    assert np.allclose(sd, np.std(simple_x, ddof=1))


def test_mov_avg(simple_x, ma_window, simple_x_mov_stats):
    x_m, x_sd, pad = mov_avg(simple_x, ma_window)

    assert np.allclose(x_m, simple_x_mov_stats[0])
    assert np.allclose(x_sd, simple_x_mov_stats[1])
    assert np.isclose(pad, 3)


def test_find_most_still():
    np.random.seed(5)
    x1 = np.random.rand(50, 2)
    np.random.seed(34)
    x2 = np.random.rand(50, 2)
    np.random.seed(13513)
    x3 = np.random.rand(50, 2)

    still_data, idx = find_most_still((x1, x2, x3), 10, return_index=True)
    result = find_most_still((x1, x2, x3), 10, return_index=False)

    assert np.allclose(np.array(still_data), np.array([[0.49851287, 0.39499911], [0.60036786, 0.45481188],
                                                       [0.18751978, 0.54213798]]))
    assert np.isclose(idx, 7)
    assert len(result) == 3  # should be 3 because thats the length of the input, which is the shape of the return
