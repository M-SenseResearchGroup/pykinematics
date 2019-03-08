"""
Methods for calculating sensor absolute and relative orientations

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import array
from numpy.linalg import norm

from .. import utility as U


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
        h_ref = U.quat_mult(q, U.quat_mult(array([0, h[0], h[1], h[2]]), U.quat_conj(q)))
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
        qDot = 0.5 * U.quat_mult(q, array([0, gyr[0], gyr[1], gyr[2]])) - self.beta * step

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
        q_dot = 0.5 * U.quat_mult(q, array([0, gyr[0], gyr[1], gyr[2]])) - self.beta * step

        # integrate to yeild quaternion
        q = q + q_dot * self.sample_period
        self.q = q / norm(q)  # normalise quaternion

        return self.q
