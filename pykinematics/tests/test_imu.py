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

        # test multiplication
        assert np.allclose(quat_mult(q1, q3), np.array([0.77195, 0.45984, 0.0711, 0.43311]), atol=1e-5)
        assert np.allclose(quat_mult(q2, q3), np.array([-0.09605, 0.7969, 0.43489, 0.40816]), atol=1e-5)
        assert np.allclose(quat_mult(q1, q2), np.array([0.5, 0.5, 0.5, 0.5]), atol=1e-5)
        assert np.allclose(quat_mult(q3, q1), np.array([0.77195, 0.45984, 0.0711, 0.43311]), atol=1e-5)
        assert np.allclose(quat_mult(q3, q2), np.array([-0.09605, 0.43489, 0.40816, 0.7969]), atol=1e-5)
        assert np.allclose(quat_mult(q1, q1), np.array([1, 0, 0, 0]), atol=1e-5)
        assert np.allclose(quat_mult(q2, q2), np.array([-0.5, 0.5, 0.5, 0.5]), atol=1e-5)
        assert np.allclose(quat_mult(q3, q3), np.array([0.19182, 0.70995, 0.10977, 0.66868]), atol=1e-5)

        # test to make sure it raises an error if an array with less than 4 elements is passed
        try:
            quat_mult(np.random.rand(3), q2)
        except ValueError:
            assert True
        else:
            assert False

        # test to make sure it raises an error if an array of multiple quaternions is passed
        try:
            quat_mult(np.random.rand(2, 4), np.random.rand(2, 4))
        except ValueError:
            assert True
        else:
            assert False

    def test_quat_conj(self):
        x1 = np.array([1, 0.5, 0.3, -0.2])
        x2 = np.array([[0.3, 0.5, -0.5, -0.8], [-0.5, 0.3, 0.1, 0.9]])

        assert np.allclose(quat_conj(x1), np.array([1, -0.5, -0.3, 0.2]))
        assert np.allclose(quat_conj(x2), np.array([[0.3, -0.5, 0.5, 0.8], [-0.5, -0.3, -0.1, -0.9]]))

    def test_quat_inv(self):
        x1 = np.array([1, 1, 1, 1])

        np.random.seed(1)
        x2 = np.random.rand(4)

        assert np.allclose(quat_inv(x1), np.array([0.25, -0.25, -0.25, -0.25]))
        assert np.allclose(quat_inv(x2), np.array([0.531793913, -0.918570667, -0.000145852811, -0.385539899]))



