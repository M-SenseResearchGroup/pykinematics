"""
Testing of functions in pykinematics/common.py
"""
from pykinematics.common import *

import numpy as np


def test_mov_avg():
    x = np.array([1, 9, 4, 2, 1, 3, 5, 4, 3, 5, 1, 3, 8, 4, 7, 6, 1, 2])

    x_m, x_sd, pad = mov_avg(x, 5)

    assert (x_m == array([0.0, 0.0, 0.0, 3.4, 3.8, 3.0, 3.0, 3.2, 4.0, 3.6, 3.2, 4.0, 4.2, 4.6, 5.6, 5.2, 0.0,
                          0.0])).all()
    assert np.allclose(x_sd, array([0.0, 0.0, 0.0, 3.36154726, 3.1144823, 1.58113883, 1.58113883, 1.4832397, 1.0,
                                    1.67332005, 1.4832397, 2.64575131, 2.58843582, 2.88097206, 2.07364414, 2.77488739,
                                    0.0, 0.0]), rtol=1e-4, atol=1e-7)
    assert np.isclose(pad, 3)
