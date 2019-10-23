"""
Testing of functions and classes for IMU based estimation of joint kinematics
"""
from pykinematics.imu.utility import *

import numpy as np


class TestImuUtility:
    def test_calc_derivative(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.0, -0.5, -0.4, -0.35, -0.45, -0.8])
        dt = 0.1

        dx2 = calc_derivative(x, dt, order=2)
        dx4 = calc_derivative(x, dt, order=4)

        assert np.allclose(dx2, array([1.0, 1.0, 1.0, 0.5, -0.5, -2.0, -4.0, -2.0, 0.75, -0.25, -2.25, -4.75]))
        assert np.allclose(dx4, array([1.25, 0.41666667, 1.75, 1.08333333, -0.41666667, -2.75, -5.33333333,
                                       -2.70833333, 0.625, -1.41666667, -0.70833333, -5.08333333]))

