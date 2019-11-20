"""
Testing of functions and classes for IMU based estimation of joint kinematics
"""
import pytest
from numpy import isclose, random, insert, array, zeros, gradient, random
import h5py

from pykinematics.imu.utility import *
from pykinematics.imu.orientation import *
from pykinematics.imu.lib.joints import *
from pykinematics.imu.lib.calibration import *
from pykinematics.imu.lib.angles import *


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
    @pytest.mark.integration
    def test_madgwick(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            acc = f_['Star Calibration']['Right Thigh']['Accelerometer'][()]
            gyr = f_['Star Calibration']['Right Thigh']['Gyroscope'][()]
            mag = f_['Star Calibration']['Right Thigh']['Magnetometer'][()]

        # initialize object
        ahrs = MadgwickAHRS(sample_period=1/128, q_init=array([1, 0, 0, 0]), beta=0.041)

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

    @pytest.mark.integration
    def test_ssro(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            acc_lu = f_['Star Calibration']['Lumbar']['Accelerometer'][()]
            gyr_lu = f_['Star Calibration']['Lumbar']['Gyroscope'][()]
            mag_lu = f_['Star Calibration']['Lumbar']['Magnetometer'][()]

            acc_rt = f_['Star Calibration']['Right Thigh']['Accelerometer'][()]
            gyr_rt = f_['Star Calibration']['Right Thigh']['Gyroscope'][()]
            mag_rt = f_['Star Calibration']['Right Thigh']['Magnetometer'][()]

        # initialize object. Set params so if defaults change don't need to re-do
        ssro = SSRO(c=0.01, N=64, error_factor=5e-8, sigma_g=1e-3, sigma_a=6e-3, grav=9.81, init_window=8)

        x_ = ssro.run(acc_lu, acc_rt, gyr_lu, gyr_rt, mag_lu, mag_rt, dt=1/128)  # run the algorithm

        assert allclose(x_[-1, :3], array([-0.98440996, -0.02239243, 0.17445807]))
        assert allclose(x_[-1, 3:6], array([-0.99828522, -0.03859549, -0.04401142]))
        assert allclose(x_[-1, 6:], array([0.76327537, -0.6417982, 0.06491315, 0.03594532]))


class TestImuJoint:
    def test_joint_center_bad_method(self):
        with pytest.raises(ValueError) as e_info:
            Center(method='not SAC or SSFC')

    def test_joint_center_bad_mask_data(self):
        with pytest.raises(ValueError) as e_info:
            Center(mask_data='not acc or gyr')

    def test_joint_center_not_enough_samples(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            acc_lu = f_['Star Calibration']['Lumbar']['Accelerometer'][()]
            gyr_lu = f_['Star Calibration']['Lumbar']['Gyroscope'][()]

            acc_rt = f_['Star Calibration']['Right Thigh']['Accelerometer'][()]
            gyr_rt = f_['Star Calibration']['Right Thigh']['Gyroscope'][()]

        dgyr_lu = gradient(gyr_lu, 1 / 128, axis=0)
        dgyr_rt = gradient(gyr_rt, 1 / 128, axis=0)

        jc_comp = Center(g=9.81, method='SAC', mask_input=True, min_samples=5000, mask_data='gyr', opt_kwargs=None)

        with pytest.raises(ValueError) as e_info:
            rlu, rrt, res = jc_comp.compute(acc_lu, acc_rt, gyr_lu * 0.001, gyr_rt * 0.001, dgyr_lu, dgyr_rt, None)

        jc_comp.method = 'SSFC'
        with pytest.raises(ValueError) as e_info:
            rlu, rrt, res = jc_comp.compute(acc_lu, acc_rt, gyr_lu * 0.001, gyr_rt * 0.001, dgyr_lu, dgyr_rt, None)

    @pytest.mark.integration
    def test_joint_center_sac(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            acc_lu = f_['Star Calibration']['Lumbar']['Accelerometer'][()]
            gyr_lu = f_['Star Calibration']['Lumbar']['Gyroscope'][()]
            mag_lu = f_['Star Calibration']['Lumbar']['Magnetometer'][()]

            acc_rt = f_['Star Calibration']['Right Thigh']['Accelerometer'][()]
            gyr_rt = f_['Star Calibration']['Right Thigh']['Gyroscope'][()]
            mag_rt = f_['Star Calibration']['Right Thigh']['Magnetometer'][()]

        dgyr_lu = gradient(gyr_lu, 1/128, axis=0)
        dgyr_rt = gradient(gyr_rt, 1/128, axis=0)

        jc_comp = Center(g=9.81, method='SAC', mask_input=True, min_samples=500, mask_data='gyr', opt_kwargs=None)

        ssro = SSRO(c=0.01, N=64, error_factor=5e-8, sigma_g=1e-3, sigma_a=6e-3, grav=9.81, init_window=8)

        x_ = ssro.run(acc_lu, acc_rt, gyr_lu, gyr_rt, mag_lu, mag_rt, dt=1 / 128)  # run the algorithm
        R = quat2matrix(x_[:, 6:])

        # mask gyr data
        rlu, rrt, res = jc_comp.compute(acc_lu, acc_rt, gyr_lu, gyr_rt, dgyr_lu, dgyr_rt, R)

        assert allclose(rlu, array([-0.05498745, -0.07630086, 0.02368247]))
        assert allclose(rrt, array([0.22075409, 0.02654856, 0.04593395]))

        # mask acc data
        jc_comp.mask_data = 'acc'
        rlu, rrt, res = jc_comp.compute(acc_lu, acc_rt, gyr_lu, gyr_rt, dgyr_lu, dgyr_rt, R)

        assert allclose(rlu, array([-0.08500494, -0.11536752, 0.06489129]))
        assert allclose(rrt, array([0.19339046, 0.02506993, 0.04676906]))

    @pytest.mark.integration
    def test_joint_center_ssfc(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            acc_lu = f_['Star Calibration']['Lumbar']['Accelerometer'][()]
            gyr_lu = f_['Star Calibration']['Lumbar']['Gyroscope'][()]

            acc_rt = f_['Star Calibration']['Right Thigh']['Accelerometer'][()]
            gyr_rt = f_['Star Calibration']['Right Thigh']['Gyroscope'][()]

        dgyr_lu = gradient(gyr_lu, 1/128, axis=0)
        dgyr_rt = gradient(gyr_rt, 1/128, axis=0)

        jc_comp = Center(g=9.81, method='SSFC', mask_input=True, min_samples=500, mask_data='gyr', opt_kwargs={})

        rlu, rrt, res = jc_comp.compute(acc_lu, acc_rt, gyr_lu, gyr_rt, dgyr_lu, dgyr_rt, None)

        assert allclose(rlu, array([-0.11404692, -0.03627268, 0.0426779]))
        assert allclose(rrt, array([0.25021739, 0.0281134, 0.06362797]))

        # test with masking acc
        jc_comp.mask_data = 'acc'
        rlu, rrt, res = jc_comp.compute(acc_lu, acc_rt, gyr_lu, gyr_rt, dgyr_lu, dgyr_rt, None)

        assert allclose(rlu, array([-0.01771183, -0.10908138,  0.02415292]))
        assert allclose(rrt, array([0.25077565, 0.02290044, 0.05807483]))

    @pytest.mark.integration
    def test_knee_axis(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            gyr_rs = f_['Star Calibration']['Right Shank']['Gyroscope'][()]

            gyr_rt = f_['Star Calibration']['Right Thigh']['Gyroscope'][()]

        ka = KneeAxis(mask_input=True, min_samples=500, opt_kwargs={})

        jrt, jrs = ka.compute(gyr_rt, gyr_rs)

        assert allclose(jrt, array([0.05436506, 0.13249331, 0.98969185]))
        assert allclose(jrs, array([-0.0713455, 0.78421354, 0.61637565]))

    @pytest.mark.parametrize(('c1', 'c2', 'c2s', 'ax'), (
            (array([0.1, -0.05, 0.03]), array([-0.11, -0.3, -0.01]), True, array([-0.01, -0.02, 0.04])),
            (array([0.1, -0.05, 0.03]), array([-0.11, -0.3, -0.01]), False, array([-0.21, 0.02, -0.04]))
    ))
    def test_fixed_axis(self, c1, c2, c2s, ax):
        axis = fixed_axis(c1, c2, c2s)


class TestImuCalibration:
    def test_get_acc_scale(self, sample_file):
        with h5py.File(sample_file, 'r') as f_:
            acc = f_['Static Calibration']['Lumbar']['Accelerometer'][()]

        scale = get_acc_scale(acc, gravity=9.81)

        assert isclose(scale, 0.9753079830416251)

    @pytest.mark.integration
    def test_static_calibration(self, sample_file, pelvis_af, left_thigh_af, right_thigh_af):
        with h5py.File(sample_file, 'r') as f_:
            stat = {}
            star = {}

            for loc in ['Left Thigh', 'Lumbar', 'Right Thigh']:
                stat[loc] = {}
                star[loc] = {}
                for meas in ['Accelerometer', 'Gyroscope', 'Magnetometer']:
                    stat[loc][meas] = f_['Static Calibration'][loc][meas][()]
                    star[loc][meas] = f_['Star Calibration'][loc][meas][()]

        ssro = SSRO(c=0.01, N=64, error_factor=5e-8, sigma_g=1e-3, sigma_a=6e-3, grav=9.84, init_window=8)

        lt_l_q = ssro.run(stat['Lumbar']['Accelerometer'], stat['Left Thigh']['Accelerometer'],
                          stat['Lumbar']['Gyroscope'], stat['Left Thigh']['Gyroscope'],
                          stat['Lumbar']['Magnetometer'], stat['Left Thigh']['Magnetometer'], dt=1/128)
        rt_l_q = ssro.run(stat['Lumbar']['Accelerometer'], stat['Right Thigh']['Accelerometer'],
                          stat['Lumbar']['Gyroscope'], stat['Right Thigh']['Gyroscope'],
                          stat['Lumbar']['Magnetometer'], stat['Right Thigh']['Magnetometer'], dt=1 / 128)

        lum_r_r = array([-0.1, -0.05, 0.05])
        lum_r_l = array([0.1, -0.5, 0.5])

        thi_r_lum_r = array([0.25, 0.2, 0.075])
        thi_r_lum_l = array([0.25, -0.15, 0.04])

        thi_r_kne_r = array([-0.18, 0.1, 0])
        thi_r_kne_l = array([-0.15, -0.15, 0.06])

        pelvis_axis = fixed_axis(lum_r_l, lum_r_r, center_to_sensor=True)
        l_thigh_axis = fixed_axis(thi_r_kne_l, thi_r_lum_l, center_to_sensor=True)
        r_thigh_axis = fixed_axis(thi_r_kne_r, thi_r_lum_r, center_to_sensor=True)

        p_AF, lt_AF, rt_AF = static(lt_l_q, rt_l_q, pelvis_axis, l_thigh_axis, r_thigh_axis,
                                    stat['Lumbar']['Gyroscope'], stat['Left Thigh']['Gyroscope'],
                                    stat['Right Thigh']['Gyroscope'], 128, window=1.0)

        assert all((allclose(i, j) for i, j in zip(p_AF, pelvis_af)))
        assert all((allclose(i, j) for i, j in zip(lt_AF, left_thigh_af)))
        assert all((allclose(i, j) for i, j in zip(rt_AF, right_thigh_af)))


class TestImuJointAngles:
    def test_hip_from_frames(self, pelvis_af, left_thigh_af, R):
        angles = hip_from_frames(pelvis_af, left_thigh_af, R, side='left', zero_angles=False)

        assert allclose(angles, array([[-61.27792272, -19.45858847, -6.21248057],
                                       [-61.08468402, -19.6292999, -11.27089438],
                                       [-60.98217794, -19.70846086, -13.0045286]]))
