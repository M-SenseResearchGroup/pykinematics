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

    def test_quat_mult(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0.5, 0.5, 0.5, 0.5])

        np.random.seed(1304913)
        q3 = np.random.rand(4)
        q3 /= np.linalg.norm(q3)

        assert np.allclose(quat_mult(q1, q3), np.array([0.77195, 0.45984, 0.0711, 0.43311]), atol=1e-5)
        assert np.allclose(quat_mult(q2, q3), np.array([-0.09605, 0.7969, 0.43489, 0.40816]), atol=1e-5)
        assert np.allclose(quat_mult(q1, q2), np.array([0.5, 0.5, 0.5, 0.5]), atol=1e-5)
        assert np.allclose(quat_mult(q3, q1), np.array([0.77195, 0.45984, 0.0711, 0.43311]), atol=1e-5)
        assert np.allclose(quat_mult(q3, q2), np.array([-0.09605, 0.43489, 0.40816, 0.7969]), atol=1e-5)
        assert np.allclose(quat_mult(q1, q1), np.array([1, 0, 0, 0]), atol=1e-5)
        assert np.allclose(quat_mult(q2, q2), np.array([-0.5, 0.5, 0.5, 0.5]), atol=1e-5)
        assert np.allclose(quat_mult(q3, q3), np.array([0.19182, 0.70995, 0.10977, 0.66868]), atol=1e-5)



