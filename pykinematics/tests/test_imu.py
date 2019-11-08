"""
Testing of functions and classes for IMU based estimation of joint kinematics
"""
import pytest
from numpy import isclose, random, insert, array, zeros
import h5py

from pykinematics.imu.utility import *
from pykinematics.imu.orientation import *


class TestImuUtility:
    def test_calc_derivative(self, F, derivative_result):
        dt = 0.1
        f = calc_derivative(F, dt, derivative_result[0])

        assert allclose(f, derivative_result[1])

    def test_quat_mult(self, qmult):
        assert allclose(quat_mult(qmult[0], qmult[1]), qmult[2])

    @pytest.mark.parametrize(('q1', 'q2'), (
            (random.rand(4), random.rand(3)),
            (random.rand(3), random.rand(3)),
            (random.rand(4, 2), random.rand(4, 2)),
            (random.rand(8), random.rand(4))))
    def test_quat_mult_error(self, q1, q2):
        with pytest.raises(ValueError) as e_info:
            quat_mult(q1, q2)

    def test_quat_conj(self, qconj):
        assert allclose(quat_conj(qconj[0]), qconj[1])

    def test_quat_inv(self, qinv):
        qn_ = qinv
        q_ = qn_ * 3

        assert allclose(quat_mult(qn_, quat_inv(qn_)), array([1, 0, 0, 0]))
        assert isclose(norm(qn_) * norm(quat_inv(qn_)), 1.0)
        assert allclose(quat_mult(q_, quat_inv(q_)), array([1, 0, 0, 0]))
        assert isclose(norm(q_) * norm(quat_inv(q_)), 1.0)

    def test_quat_inv_error(self):
        q = random.rand(2, 4)
        with pytest.raises(Exception):
            quat_inv(q)

    def test_quat2matrix(self, q2mat):
        assert allclose(quat2matrix(q2mat[0]), q2mat[1])

    @pytest.mark.parametrize(('q', 'result'), (
            (array([[1, 0, 0, 0], [0.26, 0.13, 0.64, -0.71]]), array([0.79481443, 0.08177961, 0.40260729, -0.44664246])),
    ))
    def test_quat_mean(self, q, result):
        assert allclose(quat_mean(q), result)

    @pytest.mark.parametrize(('v1', 'v2'), ((random.rand(3), random.rand(3)),
                                            (array([-0.8, .044, 1.34]), array([-0.8, .044, 1.34]))))
    def test_vec2quat(self, v1, v2):
        v1 /= norm(v1)
        v2 /= norm(v2)

        q = vec2quat(v1, v2)
        v2_comp = quat_mult(quat_mult(q, insert(v1, 0, 0)), quat_inv(q))[1:]
        v1_comp = quat_mult(quat_mult(quat_inv(q), insert(v2, 0, 0)), q)[1:]

        assert allclose(v2, v2_comp)
        assert allclose(v1, v1_comp)


class TestImuOrientation:
    def test_madgwick(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            acc = f_['Star Calibration']['Right Thigh']['Accelerometer'][()]
            gyr = f_['Star Calibration']['Right Thigh']['Gyroscope'][()]
            mag = f_['Star Calibration']['Right Thigh']['Magnetometer'][()]

        # initialize object
        ahrs = MadgwickAHRS(sample_period=1/128, q_init=array([1, 0, 0, 0]))

        for i in range(150):  # only run for 150 samples, not need to waste time processing the whole trial
            q_mimu = ahrs.update(gyr[i], acc[i], mag[i])
            q_imu = ahrs.updateIMU(gyr[i], acc[i])

        assert allclose(q_mimu, array([0.99532435, -0.00598765, 0.09583466, 0.01045504]))
        assert allclose(q_imu, array([0.99529476, -0.00603338, 0.09613962, 0.0104455]))

    def test_ssro_error(self):
        with pytest.raises(ValueError) as e_info:
            SSRO(c=1.01)
        with pytest.raises(ValueError) as e_info:
            SSRO(c=-0.1)




