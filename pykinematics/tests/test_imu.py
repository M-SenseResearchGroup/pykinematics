"""
Testing of functions and classes for IMU based estimation of joint kinematics
"""
import pytest
from numpy import array, allclose, isclose, random

from pykinematics.imu.utility import *


class TestImuUtility:
    @pytest.mark.parametrize(('x', 'order', 'result'), (
            (array([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.0, -0.5, -0.4, -0.35, -0.45, -0.8]), 2,
             array([1.0, 1.0, 1.0, 0.5, -0.5, -2.0, -4.0, -2.0, 0.75, -0.25, -2.25, -4.75])),
            (array([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.0, -0.5, -0.4, -0.35, -0.45, -0.8]), 4,
             array([1.25, 0.41666667, 1.75, 1.08333333, -0.41666667, -2.75, -5.33333333, -2.70833333, 0.625,
                    -1.41666667, -0.70833333, -5.08333333]))))
    def test_calc_derivative(self, x, order, result):
        dt = 0.1
        dx = calc_derivative(x, dt, order=order)

        assert allclose(dx, result)

    @pytest.mark.parametrize(('q1', 'q2', 'result'), (
            (array([1, 0, 0, 0]), array([0.5, 0.5, 0.5, 0.5]), array([0.5, 0.5, 0.5, 0.5])),
            (array([1, 0, 0, 0]), array([0.77, 0.46, 0.07, 0.43]), array([0.77, 0.46, 0.07, 0.43])),
            (array([0.5, 0.5, 0.5, 0.5]), array([0.77, 0.46, 0.07, 0.43]), array([-0.095, 0.795, 0.435, 0.405])),
            (array([0.77, 0.46, 0.07, 0.43]), array([0.77, 0.46, 0.07, 0.43]), array([0.1915, 0.7084, 0.1078, 0.6622])),
    ))
    def test_quat_mult(self, q1, q2, result):
        assert allclose(quat_mult(q1, q2), result)

    @pytest.mark.parametrize(('q1', 'q2'), (
            (array([0.77, 0.46, 0.07, 0.43]), array([1.3072, -0.455, 0.0312])),
            (array([1.3072, -0.455, 0.0312]), array([0.77, 0.46, 0.07, 0.43])),
            (random.rand(4, 2), random.rand(4, 2)),
            (random.rand(8), random.rand(8))))
    def test_quat_mult_error(self, q1, q2):
        with pytest.raises(ValueError) as e_info:
            quat_mult(q1, q2)

    @pytest.mark.parametrize(('q', 'result'), (
            (array([1, 0.5, 0.3, -0.2]), array([1, -0.5, -0.3, 0.2])),
            (array([[0.3, 0.5, -0.5, -0.8], [-0.5, 0.3, 0.1, 0.9]]), array([[0.3, -0.5, 0.5, 0.8],
                                                                            [-0.5, -0.3, -0.1, -0.9]]))))
    def test_quat_conj(self, q, result):
        assert allclose(quat_conj(q), result)

    @pytest.mark.parametrize(('q', ), (
            (array([1, 1, 1, 1]),),
            (random.rand(4),)))
    def test_quat_inv(self, q):
        qn = q / norm(q)

        qinv = quat_inv(q)
        qninv = quat_inv(qn)

        assert allclose(quat_mult(q, qinv), array([1, 0, 0, 0]))
        assert allclose(quat_mult(qn, qninv), array([1, 0, 0, 0]))
        assert isclose(norm(qninv), 1.0)

    def test_quat_inv_error(self):
        q = random.rand(2, 4)
        with pytest.raises(Exception):
            quat_inv(q)


