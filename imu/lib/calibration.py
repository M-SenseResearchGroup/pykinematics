"""
Methods for calibration of imu sensors for use in joint angle calculation

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import mean, ceil, cross
from numpy.linalg import norm

from pymotion.imu import utility
from pymotion import common


__all__ = ['get_acc_scale', 'process_static_calibration']


def get_acc_scale(acc, gravity=9.81):
    """
    Get the acceleration scale factor so that the mean of the acceleration during the static trial is equal between
    sensors.

    Parameters
    ----------
    acc : numpy.ndarray
        Nx3 array of accelerations during a static calibration trial.
    gravity : float, optional
        Value of local gravitational acceleration. Default is 9.81m/s^2

    Returns
    -------
    scale : float
        Scale factor to multiply acceleration by
    """
    mean_acc = mean(norm(acc, axis=1))

    return gravity / mean_acc


def process_static_calibration(lt_p_q, rt_p_q, pelvis_axis, l_thigh_axis, r_thigh_axis, pelvis_w, l_thigh_w, r_thigh_w,
                               fs, window=1.0):
    """
    Process a static standing data to create the anatomical axes for the pelvis and thighs.

    Parameters
    ----------
    lt_p_q : numpy.ndarray
        Nx4 array of quaternions representing the rotation from the left thigh to the pelvis.
    rt_p_q : numpy.ndarray
        Nx4 array of quaternions representing the rotation from the right thigh to the pelvis.
    pelvis_axis : numpy.ndarray
        Pelvis fixed axis.
    l_thigh_axis : numpy.ndarray
        Left thigh fixed axis.
    r_thigh_axis : numpy.ndarray
        Right thigh fixed axis.
    pelvis_w : numpy.ndarray
        Nx3 array of angular velocities measured by the pelvis sensor.
    l_thigh_w : numpy.ndarray
        Nx3 array of angular velocities measured by the left thigh sensor.
    r_thigh_w : numpy.ndarray
        Nx3 array of angular velocities measured by the right thigh sensor.
    fs : float
        Sampling frequency of the sensors.
    window : float, optional
        Length of time for the most still period. Default is 1.0s.

    Returns
    -------
    pelvis_AF : tuple
        Tuple of the x, y, and z axes of the pelvis anatomical frame.
    l_thigh_AF : tuple
        Tuple of the x, y, and z axes of the left thigh anatomical frame.
    r_thigh_AF : tuple
        Tuple of the x, y, and z axes of the right thigh anatomical frame.
    """

    # find the most still period in the angular velocity data of the pelvis and thighs
    _, ind = common.find_most_still((pelvis_w, l_thigh_w, r_thigh_w), int(window * fs), return_index=True)

    pad = int(ceil(window * fs / 2))

    # compute the rotations during the most still time
    if (ind - pad) < 0:
        l_q_mean = utility.quat_mean(lt_p_q[:ind + pad, :])
        r_q_mean = utility.quat_mean(rt_p_q[:ind + pad, :])
    else:
        l_q_mean = utility.quat_mean(lt_p_q[ind - pad:ind + pad, :])
        r_q_mean = utility.quat_mean(rt_p_q[ind - pad:ind + pad, :])

    lt_p_R = utility.quat2matrix(l_q_mean)
    rt_p_R = utility.quat2matrix(r_q_mean)

    # compute the left and right hip coordinate systems
    l_e1 = pelvis_axis
    l_e3 = lt_p_R @ l_thigh_axis
    l_e2 = cross(l_e3, l_e1)
    l_e2 /= norm(l_e2)

    r_e1 = pelvis_axis
    r_e3 = rt_p_R @ r_thigh_axis
    r_e2 = cross(r_e3, r_e1)
    r_e2 /= norm(r_e2)

    # form the pelvis anatomical axes
    p_z = pelvis_axis
    p_x = (r_e2 + l_e2) / 2
    p_y = cross(p_z, p_x)
    p_y /= norm(p_y)
    p_x = cross(p_y, p_z)
    p_x /= norm(p_x)

    # form the left thigh anatomical axes
    lt_x = lt_p_R.T @ l_e2
    lt_y = l_thigh_axis
    lt_z = cross(lt_x, lt_y)
    lt_z /= norm(lt_z)

    # form the right thigh anatomical axes
    rt_x = rt_p_R.T @ r_e2
    rt_y = r_thigh_axis
    rt_z = cross(rt_x, rt_y)
    rt_z /= norm(rt_z)

    # create the anatomical frames
    pelvis_AF = (p_x, p_y, p_z)
    l_thigh_AF = (lt_x, lt_y, lt_z)
    r_thigh_AF = (rt_x, rt_y, rt_z)

    return pelvis_AF, l_thigh_AF, r_thigh_AF
