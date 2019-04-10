"""
Methods for calculating sensor absolute and relative orientations

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import array, zeros, cross, sqrt, abs as nabs, arccos, sin, mean, identity, sum
from numpy.linalg import norm

from pymotion.imu import utility
from pymotion.imu.optimize import UnscentedKalmanFilter


__all__ = ['MadgwickAHRS', 'OrientationComplementaryFilter', 'SROFilter']


class MadgwickAHRS:
    def __init__(self, sample_period=1/256, q_init=array([1, 0, 0, 0]), beta=0.041):
        """
        Algorithm for estimating the orientation of an inertial sensor with or without a magnetometer.

        Parameters
        ----------
        sample_period : float, optional
            Sampling period in seconds.  Default is 1/256 (ie sampling frequency is 256Hz).
        q_init : numpy.ndarray, optional
            Initial quaternion estimate.  Default is [1, 0, 0, 0].
        beta : float, optional
            Beta value for the algorithm.  Default is 1.0

        References
        ---------
        S. Madgwick et al. "Estimation of IMU and MARG orientation using a gradient descent algorith." IEEE Intl. Conf. on
        Rehab. Robotics. 2011.
        """
        self.sample_period = sample_period
        self.q = q_init
        self.beta = beta

    def update(self, gyr, acc, mag):
        """
        Perform update step.

        Parameters
        ----------
        gyr : numpy.ndarray
            Angular velocity at time t.  Units of rad/s.
        acc : numpy.ndarray
            Acceleration at time t.  Units of g.
        mag : numpy.ndarray
            Magnetometer reading at time t.

        Returns
        -------
        q : numpy.ndarray
            Quaternion estimate of orientation at time t.
        """
        # short name for the quaternion
        q = self.q

        # normalize accelerometer measurement
        a = acc / norm(acc)

        # normalize magnetometer measurement
        h = mag / norm(mag)

        # reference direction of earth's magnetic field
        h_ref = utility.quat_mult(q, utility.quat_mult(array([0, h[0], h[1], h[2]]), utility.quat_conj(q)))
        b = array([0, norm(h_ref[1:3]), 0, h_ref[3]])

        # Gradient Descent algorithm corrective step
        F = array([2 * (q[1] * q[3] - q[0] * q[2]) - a[0],
                   2 * (q[0] * q[1] + q[2] * q[3]) - a[1],
                   2 * (0.5 - q[1]**2 - q[2]**2) - a[2],
                   2 * b[1] * (0.5 - q[2]**2 - q[3]**2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - h[0],
                   2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - h[1],
                   2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1]**2 - q[2]**2) - h[2]])
        J = array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                   [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                   [0, -4 * q[1], -4 * q[2], 0],
                   [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0], -4 * b[1] * q[3] + 2 * b[3] * q[1]],
                   [-2 * (b[1] * q[3] - b[3] * q[1]), 2 * (b[1] * q[2] + b[3] * q[0]), 2 * (b[1] * q[1] + b[3] * q[3]), -2 * (b[1] * q[0] - b[3] * q[2])],
                   [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2], 2 * b[1] * q[1]]])

        step = J.T @ F
        step /= norm(step)  # normalize step magnitude

        # compute rate of change of quaternion
        qDot = 0.5 * utility.quat_mult(q, array([0, gyr[0], gyr[1], gyr[2]])) - self.beta * step

        # integrate to yeild quaternion
        self.q = q + qDot * self.sample_period
        self.q /= norm(self.q)

        return self.q

    def updateIMU(self, gyr, acc):
        """
        Perform update step using only gyroscope and accelerometer measurements

        Parameters
        ----------
        gyr : numpy.ndarray
            Angular velocity at time t.  Units of rad/s.
        acc : numpy.ndarray
            Acceleration at time t.  Units of g.

        Returns
        -------
        q : numpy.ndarray
            Quaternion estimate of orientation at time t.
        """
        a = acc / norm(acc)  # normalize accelerometer magnitude
        q = self.q  # short name
        # gradient descent algorithm corrective step
        F = array([2 * (q[1] * q[3] - q[0] * q[2]) - a[0],
                   2 * (q[0] * q[1] + q[2] * q[3]) - a[1],
                   2 * (0.5 - q[1]**2 - q[2]**2) - a[2]])
        J = array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                   [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                   [0, -4 * q[1], -4 * q[2], 0]])
        step = J.T @ F
        step /= norm(step)  # normalise step magnitude

        # compute rate of change quaternion
        q_dot = 0.5 * utility.quat_mult(q, array([0, gyr[0], gyr[1], gyr[2]])) - self.beta * step

        # integrate to yeild quaternion
        q = q + q_dot * self.sample_period
        self.q = q / norm(q)  # normalise quaternion

        return self.q


class OrientationComplementaryFilter:
    def __init__(self, g=9.81, alpha=0.1, beta=0.1, compute_bias=True, bias_alpha=0.1, acc_thresh=0.1, gyr_thresh=0.2,
                 delta_gyr_thresh=0.01, adaptive_gain=True, gain_error_thresh=(0.1, 0.2), interp_thresh=0.9,
                 acc_corr=True):
        """
        Complementary filter for orientation estimation.

        Parameters
        ----------
        g : float, optional
            Local value of gravitational acceleration. Default is 9.8 m/s^2.
        alpha : float, optional
            Acceleration complementary filter cut-off frequency.  Default is 0.1.
        beta : float, optional
            Magnetic field complementary filter cut-off frequency. Default is 0.1.
        compute_bias : bool, optional
            Compute the bias of the angular velocity. Default is True.
        bias_alpha : float, optional
            Angular velocity bias gain.  Default is 0.1
        acc_thresh : float, optional
            Threshold for determining still periods from acceleration for bias updating. Default is 0.1.
        gyr_thresh : float, optional
            Threshold for determining still periods from angular velocity for bias updating. Default is 0.2
        delta_gyr_thresh : float, optional
            Threshold for determining still periods from change since previous update of angular velocity for
            bias updating.  Default is 0.01.
        adaptive_gain : bool, optional
            Use adaptive gain or not.  Default is True.
        gain_error_thresh : array_like, optional
            Cutoff values for gain factor calculation. Default is (0.1, 0.2).
        interp_thresh : float, optional
            Below which value to use SLERP vs LERP. Default is 0.9
        acc_corr : bool, optional
            Correct acceleration by removing rotational acceleration. Default is True

        References
        ----------
        Valenti et al. Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMU and MARGs. Sensors. 2015
        """
        self.gravity = g

        self.alpha = alpha
        self.beta = beta

        self.comp_bias = compute_bias
        self.bias_alpha = bias_alpha

        self.acc_thresh = acc_thresh
        self.w_thresh = gyr_thresh
        self.delta_w_thresh = delta_gyr_thresh

        self.ada_gain = adaptive_gain
        self.ada_thresh1 = gain_error_thresh[0]
        self.ada_thresh2 = gain_error_thresh[1]
        # setup slope and intercept for adaptive gain factor calculation
        self.ada_m = 1 / (self.ada_thresh1 - self.ada_thresh2)
        self.ada_b = 1 - self.ada_m * self.ada_thresh1

        self.interp_thresh = interp_thresh

        self.acc_corr = acc_corr
        # initialize angular velocity bias
        self.w_bias = zeros(3)

    def run(self, acc, gyr, mag, dt, r):
        """
        Run the orientation estimation over the data

        Parameters
        ----------
        acc : numpy.ndarray
            Nx3 array of accelerations.
        gyr : numpy.ndarray
            Nx3 array of angular velocities.
        mag : numpy.ndarray
            Nx3 array of magnetic field readings.
        dt : float
            Sampling time.

        Attributes
        ----------
        q_ : numpy.ndarray
            Nx4 array of orientation estimates from the global frame to the local frame.
        """
        # store input values
        if self.acc_corr:
            corr = cross(gyr, cross(gyr, r))
            self.a = acc - corr
            # store the acceleration norm since its useful
            self.a_mag = norm(self.a, axis=1)

            self.a /= norm(self.a, axis=1, keepdims=True)
        else:
            # store the acceleration norm since its useful
            self.a_mag = norm(acc, axis=1)
            self.a = acc / self.a_mag.reshape((-1, 1))
        self.w = gyr.copy()
        self.h = mag / norm(mag, axis=1, keepdims=True)
        self.dt = dt

        # get the first estimate of q
        self.q = self.get_measurement(0)

        # get the number of elements
        n = self.a_mag.size

        # storage for quaternion estimate
        self.q_ = zeros((n, 4))

        # run estimation procedure
        for i in range(n):
            # update the bias if doing so
            if self.comp_bias:
                self.update_biases(i)

            # get the prediction for the state
            q_pred = self.get_prediction(i)

            # get the acceleration based correction
            dq_acc = self.get_acc_correction(q_pred, i)

            # get the adaptive gain factor.  Factor is 1 if not using an adaptive gain
            factor = self.get_adaptive_gain_factor(i)
            alpha = self.alpha * factor

            # interpolate the acceleration based correction
            dq_acc_int = self.get_scaled_quaternion(dq_acc, alpha)

            q_pred_acorr = utility.quat_mult(q_pred, dq_acc_int)

            # get the magnetometer based correction
            dq_mag = self.get_mag_correction(q_pred_acorr, i)

            # interpolate the magnetometer based correction
            dq_mag_int = self.get_scaled_quaternion(dq_mag, self.beta)

            # correct the prediction resulting in final estimate
            self.q = utility.quat_mult(q_pred_acorr, dq_mag_int)

            # normalize estimate
            self.q /= norm(self.q)

            # save the orientation estimate
            self.q_[i] = self.q

    def get_measurement(self, ind):
        """
        Get the initial measurement guess for the algorithm.
        """
        if self.a[ind, 2] >= 0:
            b = sqrt(self.a[ind, 2] + 1)
            q_acc = array([b / sqrt(2), -self.a[ind, 1] / (sqrt(2) * b), self.a[ind, 0] / (sqrt(2) * b), 0])
        else:
            b = sqrt(1 - self.a[ind, 2])
            q_acc = array([-self.a[ind, 1] / (sqrt(2) * b), b / sqrt(2), 0, self.a[ind, 0] / (sqrt(2) * b)])

        l = utility.quat2matrix(q_acc).T @ self.h[ind]
        Gamma = l[0]**2 + l[1]**2

        if l[0] >= 0:
            b = sqrt(Gamma + l[0] * sqrt(Gamma))
            q_mag = array([b / sqrt(2 * Gamma), 0, 0, l[1] / (sqrt(2) * b)])
        else:
            b = sqrt(Gamma - l[0] * sqrt(Gamma))
            q_mag = array([l[1] / (sqrt(2) * b), 0, 0, b / sqrt(2 * Gamma)])

        q_meas = utility.quat_mult(q_acc, q_mag)
        return q_meas

    def get_state(self, ind):
        """
        Check whether or not a time point is in steady state.

        Parameters
        ----------
        ind : int
            Index of data to check

        Returns
        -------
        steady_state : bool
            Whether or not the sensor is in a steady state.
        """
        if abs(self.a_mag[ind] - self.gravity) > self.acc_thresh:
            return False
        elif (nabs(self.w[ind] - self.w[ind-1]) > self.delta_w_thresh).any():
            return False
        elif (nabs(self.w[ind] - self.w_bias) > self.w_thresh).any():
            return False
        else:
            return True

    def update_biases(self, ind):
        """
        Update the bias parameters if in steady state.

        Parameters
        ----------
        ind : int
            Index of data to update from
        """
        steady_state = self.get_state(ind)

        if steady_state:
            self.w_bias += self.bias_alpha * (self.w[ind] - self.w_bias)

    def get_prediction(self, ind):
        """
        Compute the predicted orientation estimate for the time point.

        Parameters
        ----------
        ind : int
            Index of data to use.

        """
        """
        # construct the derivative calculation matrix
        Omega = zeros((4, 4))
        Omega[0, 1:] = self.w[ind] - self.w_bias
        Omega[1:, 0] = -self.w[ind] - self.w_bias
        Omega[1:, 1:] = OrientationComplementaryFilter.skew_symmetric(self.w[ind] - self.w_bias)

        # compute the predicted orientation estimate
        q_pred = self.q + Omega @ self.q * self.dt
        """
        wu = self.w[ind] - self.w_bias

        q_pred = self.q.copy()
        q_pred[0] += 0.5 * self.dt * (wu[0] * self.q[1] + wu[1] * self.q[2] + wu[2] * self.q[3])
        q_pred[1] += 0.5 * self.dt * (-wu[0] * self.q[0] - wu[1] * self.q[3] + wu[2] * self.q[2])
        q_pred[2] += 0.5 * self.dt * (wu[0] * self.q[3] - wu[1] * self.q[0] - wu[2] * self.q[1])
        q_pred[3] += 0.5 * self.dt * (-wu[0] * self.q[2] + wu[1] * self.q[1] - wu[2] * self.q[0])

        return q_pred

    def get_acc_correction(self, q_pred, ind):
        """
        Compute the acceleration based quaternion correction for the predicted value.

        Parameters
        ----------
        ind : int
            Index to use for computation.

        Returns
        -------
        dq : numpy.ndarray
            Quaternion correction.
        """
        # compute the predicted gravity vector
        gp = utility.quat2matrix(utility.quat_inv(q_pred)) @ self.a[ind]
        gp /= norm(gp)
        # compute the correction quaternion
        # b = sqrt(gp[2] + 1)
        # dq = array([b / sqrt(2), -gp[1] / (sqrt(2) * b), gp[0] / (sqrt(2) * b), 0])
        dq = zeros(4)
        dq[0] = sqrt((gp[2] + 1) * 0.5)
        dq[1] = -gp[1] / (2 * dq[0])
        dq[2] = gp[0] / (2 * dq[0])
        return dq / norm(dq)

    def get_mag_correction(self, q_pred, ind):
        """
        Compute the magnetometer based correction for the predicted orientation.

        Parameters
        ----------
        q_pred : numpy.ndarray
            Predicted orientation estimate
        ind : int
            Index to compute correction at.

        Returns
        -------
        dq : numpy.ndarray
            Quaternion correction
        """
        # rotate the magnetic field vector into the current orientation estimate
        l = utility.quat2matrix(utility.quat_inv(q_pred)) @ self.h[ind]
        l /= norm(l)
        Gamma = l[0]**2 + l[1]**2

        # b = sqrt(Gamma + l[0] * sqrt(Gamma))
        # dq = array([b / sqrt(2 * Gamma), 0, 0, l[1] / (sqrt(2) * b)])

        beta = sqrt(Gamma + l[0] * sqrt(Gamma))
        dq = zeros(4)
        dq[0] = beta / sqrt(2 * Gamma)
        dq[3] = l[1] / (sqrt(2) * beta)

        return dq / norm(dq)

    def get_scaled_quaternion(self, q, gain, p=array([1, 0, 0, 0])):
        """
        Scale the quaternion via linear interoplation or spherical linear interpolation.

        Parameters
        ----------
        q : numpy.ndarray
            Quaternion to scale.
        p : numpy.ndarray
            Quaternion to scale towards.

        Returns
        -------
        qs : numpy.ndarray
            Scaled quaternion.
        """
        angle = arccos(sum(q * p))
        if q[0] < self.interp_thresh:
            qs = sin((1 - gain) * angle) / sin(angle) * p + sin(gain * angle) / sin(angle) * q
        else:
            qs = (1 - gain) * p + gain * q

        return qs / norm(qs)

    def get_adaptive_gain_factor(self, ind):
        """
        Get the adaptive gain factor.

        Parameters
        ----------
        ind : int
            Index to use for gain computation.

        Returns
        -------
        factor : float
            Gain factor.
        """
        factor = 1
        if not self.ada_gain:
            return factor
        else:
            err = abs(self.a_mag[ind] - self.gravity) / self.gravity
            if self.ada_thresh1 < err <= self.ada_thresh2:
                factor = self.ada_m * err + self.ada_b
            elif err > self.ada_thresh2:
                factor = 0
            return factor

    @staticmethod
    def skew_symmetric(x):
        """
        Get the skew symmetric representation of a vector.

        Parameters
        ----------
        x : numpy.ndarray
            1D vector in 3D space (ie 3 component vector)

        Returns
        -------
        X : numpy.ndarray
            3x3 skew symmetric matrix representation of x
        """
        return array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


class SROFilter:
    def __init__(self, Q=identity(4)*0.1, error_mode='linear', error_factor=0.1, g=9.81, s1_ahrs_beta=0.041,
                 s2_ahrs_beta=0.041, init_window=8, smooth_quaternions=False):
        """
        Sensor Relative Orientation Filter for determining the relative orientation between two sensors.

        Parameters
        ----------
        Q : numpy.ndarray, float, optional
            Process covariance matrix or float. Default is I * 0.1.
        error_mode : {'linear', 'exp'}, optional
            How the error is calculated, either a linear multiplication by the error_factor, or using the error raised
            to the power error_factor.
        error_factor : float, optional
            Factor effecting the error that is used for adapting the measurement covariance. Default is 0.1
        g : float, optional
            Value of local gravitational acceleration. Default is 9.81 m/s^2.
        s1_ahrs_beta : float, optional
            Beta value for Madgwicks AHRS orientation estimation for the first sensor. Default is 0.041.
        s2_ahrs_beta : float, optional
            Beta value for Madgwicks AHRS orientation estimation for the second sensor. Default is 0.041.
        init_window : int, optional
            Number of samples to average at the beginning of the trial for the initial guess for gravity vector.
            Default is 8.
        smooth_quaternions : bool, optional
            Smooth the resulting rotation quaternions. Default is False.
        """
        self.Q = Q
        self.error_mode = error_mode
        self.error_factor = error_factor
        self.g = g
        self.s1_ahrs_beta = s1_ahrs_beta
        self.s2_ahrs_beta = s2_ahrs_beta
        self.init_window = init_window
        self.smooth = smooth_quaternions

    def run(self, s1_a, s2_a, s1_w, s2_w, s1_h, s2_h, dt):
        """
        Run the filter on the data from the joint.

        Parameters
        ----------
        s1_a : numpy.ndarray
            Nx3 array of accelerations from the first sensor.
        s2_a : numpy.ndarray
            Nx3 array of accelerations from the second sensor
        s1_w : numpy.ndarray
            Nx3 array of angular velocities from the first sensor.
        s2_w : numpy.ndarray
            Nx3 array of angular velocities from the second sensor.
        s1_h : numpy.ndarray
            Nx3 array of magnetometer readings from the first sensor.
        s2_h : numpy.ndarray
            Nx3 array of magnetometer readings from the second sensor.
        dt : float
            Sampling period in seconds.

        Returns
        -------
        q : numpy.ndarray
            Nx4 array of quaternions representing the rotation from sensor 2 to sensor 1.
        """
        # number of elements
        n = s1_a.shape[0]

        # get the mean of the first few samples of acceleration which will be used to compute an initial guess for the
        # gravity vector esimation.
        s1_a0 = mean(s1_a[:self.init_window], axis=0)
        s2_a0 = mean(s2_a[:self.init_window], axis=0)

        # get the rotation from the initial accelerations to the global gravity vector
        s1_q_init = utility.vec2quat(s1_a0, array([0, 0, 1]))
        s2_q_init = utility.vec2quat(s2_a0, array([0, 0, 1]))

        # setup the AHRS algorithms
        s1_ahrs = MadgwickAHRS(dt, q_init=s1_q_init, beta=self.s1_ahrs_beta)
        s2_ahrs = MadgwickAHRS(dt, q_init=s2_q_init, beta=self.s2_ahrs_beta)

        # run the algorithms
        s1_q = zeros((n, 4))
        s2_q = zeros((n, 4))
        for i in range(n):
            s1_q[i] = s1_ahrs.updateIMU(s1_w[i], s1_a[i] / self.g)
            s2_q[i] = s2_ahrs.updateIMU(s2_w[i], s2_a[i] / self.g)

        # get the gravity axis from the rotation matrices
        s1_z = utility.quat2matrix(s1_q)[:, 2, :]
        s2_z = utility.quat2matrix(s2_q)[:, 2, :]

        # get the magnetometer readings that are not in the direction of gravity
        s1_m = s1_h - sum(s1_h * s1_z, axis=1, keepdims=True) * s1_z
        s2_m = s2_h - sum(s2_h * s2_z, axis=1, keepdims=True) * s2_z

        # get the initial guess for the orientation.  Used to initialize the UKF
        # Initial guess from aligning gravity axes
        q_z_init = utility.vec2quat(s2_z[0], s1_z[0])
        # Initial guess from aligning non-gravity vector magnetometer readings
        q_m_init = utility.vec2quat(utility.quat2matrix(q_z_init) @ s2_m[0], s1_m[0])

        q_init = utility.quat_mult(q_m_init, q_z_init)

        # Initial Kalman Filter parameters
        P_init = identity(4) * 0.1  # assume our initial guess is fairly good
        R = identity(4)  # Measurement will have some error

        ukf = UnscentedKalmanFilter(q_init, P_init, SROFilter._F, SROFilter._H, self.Q, R)

        self.q_ = zeros((n, 4))  # storage for ukf output
        self.q_[0] = q_init  # store the initial guess
        self.err = zeros(n)
        for i in range(1, n):
            # get the measurement guess for the orientation
            q_z = utility.vec2quat(s2_z[i], s1_z[i])
            q_m = utility.vec2quat(utility.quat2matrix(q_z) @ s2_m[i], s1_m[i])

            q = utility.quat_mult(q_m, q_z).reshape((4, 1))

            # get an estimate of the measurement error, based on how dynamic the motion is
            err = abs(norm(s1_a[i]) - self.g) + abs(norm(s2_a[i]) - self.g)
            if self.error_mode == 'linear':
                err *= self.error_factor
            elif self.error_mode == 'exp' or self.error_mode == 'exponential':
                err = err ** self.error_factor
            else:
                raise ValueError('error_mode must either be linear or exp (exponential)')
            self.err[i] = err
            # update the measurement noise matrix based upon this error estimate
            ukf.R = identity(4) * err

            self.q_[i] = ukf.run(q, f_kwargs=dict(s1_w=s1_w[i], s2_w=s2_w[i], dt=dt)).flatten()
            self.q_[i] /= norm(self.q_[i])  # output isn't necessarily normalized as rotation quaternions need to be

        if self.smooth:
            q = self.q_.copy()
            pad = 16
            for i in range(pad, self.q_.shape[0]-pad):
                self.q_[i] = utility.quat_mean(q[i - pad:i + pad, :])

        return self.q_

    @staticmethod
    def _F(x, s1_w, s2_w, dt):
        """
        State update function for the relative orientation unscented kalman filter.

        Parameters
        ----------
        x : numpy.ndarray
            4x9 array of state vector sigma points.
        s1_w : numpy.ndarray
            Proximal sensor angular velocity at a time point.
        s2_w : numpy.ndarray
            Distal sensor angular velocity at a time point.
        dt : float
            Sampling time difference.

        Returns
        -------
        xp : numpy.ndarray
            4x9 array of predicted state vector sigma points
        """
        xp = zeros(x.shape)
        rot_q = zeros(4)
        # iterate over the sigma point variations of the state vector
        for i in range(x.shape[1]):
            # rotate the proximal angular velocity into the distal frame
            s2_s1_w = utility.quat2matrix(x[:, i]).T @ s1_w

            # create the angular velocity quaternion which is used in the quaternion gradient calculation. The relative
            # angular velocity is distal - proximal as the relative orientation is in that direction.
            rot_q[1:] = s2_w - s2_s1_w  # rot_q = [0, wx, wy, wz]

            # calculate the quaternion gradient
            x_grad = utility.quat_mult(x[:, i], rot_q)
            xp[:, i] = x[:, i] + 0.5 * dt * x_grad

        return xp

    @staticmethod
    def _H(x):
        """
        Measurement prediction function.

        Parameters
        ----------
        x : numpy.ndarray
            4x9 array of state vector sigma points.

        Returns
        -------
        zp : numpy.ndarray
            4x9 array of predicted measured values.
        """
        # measurement is the orientation quaternion, so just return the state vector
        return x

