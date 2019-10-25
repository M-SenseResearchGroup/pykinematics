"""
Methods for calculating joint angles using previously computed joint parameters and calibrated frames

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import cross, abs as nabs, arctan2 as atan2, sum, arccos, pi, stack
from numpy.linalg import norm
from scipy.integrate import cumtrapz


__all__ = ['hip_from_frames']


def hip_from_frames(pelvis_AF, thigh_AF, R, side, zero_angles=False):
    """
    Compute the hip joint angles from the segment fixed axes computed during the calibration.

    Parameters
    ----------
    pelvis_AF : tuple
        Tuple of the x, y, and z axes of the pelvis anatomical frame.
    thigh_AF : tuple
        Tuple of the x, y, and z axes of the thigh anatomical frame.
    R : numpy.ndarray
        Nx3x3 array of rotation matrices from the thigh sensor frame to the pelvis sensor frame.
    side : {'left', 'right'}
        Side the angles are being computed for.
    zero_angles : bool, optional
        Remove any offset from zero at the start of the angles. Default is False.

    Returns
    -------
    angles : numpy.ndarray
        Nx3 array of hip angles in degrees, with the first column being Flexion - Extension,
        second column being Ad / Abduction, and the third column being Internal - External Rotation.
    """
    # get pelvis anatomical frame
    X = pelvis_AF[0]
    Y = pelvis_AF[1]
    Z = pelvis_AF[2]

    # get the thigh anatomical frame and rotate into pelvis frame
    x = R @ thigh_AF[0]
    y = R @ thigh_AF[1]
    z = R @ thigh_AF[2]

    # form the joint axes
    e1 = Z.copy()
    e3 = y.copy()

    e2 = cross(e3, e1)
    e2 /= norm(e2, axis=1, keepdims=True)

    # compute angles
    sgn = sum(cross(X, e2) * Z, axis=1)
    sgn /= nabs(sgn)
    fe = atan2(sgn * norm(cross(X, e2), axis=1), sum(X * e2, axis=1))

    if side == 'right':
        # ad / abduction calculation
        aa = -pi / 2 + arccos(sum(e1 * e3, axis=1))

        # internal - external rotation sign calculation
        sgn = sum(cross(x, e2) * -y, axis=1)
    elif side == 'left':
        aa = pi / 2 - arccos(sum(e1 * e3, axis=1))
        sgn = sum(cross(x, e2) * y, axis=1)
    else:
        raise ValueError("side must be either 'left' or 'right'.")

    sgn /= nabs(sgn)
    ier = atan2(sgn * norm(cross(x, e2), axis=1), sum(x * e2, axis=1))

    angles = stack((fe, aa, ier), axis=1) * 180 / pi
    if zero_angles:
        angles -= angles[0, :]

    return angles


def hip_from_gyr(pelvis_w, thigh_w, pelvis_axis, thigh_axis, R, side):
    """
    Compute hip angles by integrating the difference between the two sensors angular velocities about the axis
    of rotation.  Typically not very noisy on higher frequencies, but on lower frequencies where it exhibits drift
    due to integration being used.

    Parameters
    ----------
    pelvis_w : numpy.ndarray
        Nx3 array of angular velocity vectors in the pelvis sensor frame.
    thigh_w : numpy.ndarray
        Nx3 array of angular velocity vectors in the thigh sensor frame.
    pelvis_axis : numpy.ndarray
        Pelvis fixed axis.
    thigh_axis : numpy.ndarray
        Thigh fixed axis.
    R : numpy.ndarray
        Nx3x3 array of rotation matrices from the thigh sensor frame to the pelvis sensor frame.
    side : {'left', 'right'}
        Side angles are being computed for.

    Returns
    -------
    angles : numpy.ndarray
        Nx3 array of joint angles for the trial.  Columns are flexion - extension, ad / abduction, and
        internal - external rotation respectively.
    """
    # get the joint rotation axes in the pelvis frame
    pelvis_e1 = pelvis_axis
    pelvis_e3 = R @ thigh_axis

    pelvis_e2 = cross(pelvis_e3, pelvis_e1)
    pelvis_e2 /= norm(pelvis_e2, axis=1, keepdims=True)

    # get the joint rotation axes in the thigh frame
    thigh_e1 = R.transpose([0, 2, 1]) @ pelvis_axis
    thigh_e3 = thigh_axis

    thigh_e2 = cross(thigh_e3, thigh_e1)
    thigh_e2 /= norm(thigh_e2, axis=1, keepdims=True)

    # compute the differences between angular rates around their respective rotation axes
    # TODO needs to be corrected still
    if side == 'left':
        fe_int = sum(pelvis_w * pelvis_e1, axis=1) - sum(thigh_w * thigh_e1, axis=1)
        aa_int = -sum(pelvis_w * pelvis_e2, axis=1) + sum(thigh_w * thigh_e2, axis=1)
        ier_int = sum(pelvis_w * pelvis_e3, axis=1) - sum(thigh_w * thigh_e3, axis=1)
    elif side == 'right':
        fe_int = sum(pelvis_w * pelvis_e1, axis=1) - sum(thigh_w * thigh_e1, axis=1)
        aa_int = sum(pelvis_w * pelvis_e2, axis=1) - sum(thigh_w * thigh_e2, axis=1)
        ier_int = -sum(pelvis_w * pelvis_e3, axis=1) + sum(thigh_w * thigh_e3, axis=1)
    else:
        raise ValueError("side must be 'left' or 'right'.")

    # integrate the differences in angular velocities to get angles.
    fe = cumtrapz(fe_int, dx=1 / 128, initial=0)
    aa = cumtrapz(aa_int, dx=1 / 128, initial=0)
    ier = cumtrapz(ier_int, dx=1 / 128, initial=0)

    angles = stack((fe, aa, ier), axis=1) * 180 / pi
    return angles
