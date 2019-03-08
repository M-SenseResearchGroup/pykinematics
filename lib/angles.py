"""
Methods for calculating joint angles using previously computed joint parameters and calibrated frames

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import cross, abs as nabs, arctan2 as atan2, sum, arccos, pi, stack
from numpy.linalg import norm


def hip_from_frames(pelvis_AF, thigh_AF, R, zero_angles=False):
    """
    Compute the hip joint angles from fixed and reference axes from the anatomical frames.

    Parameters
    ----------
    hip_AF : numpy.ndarray
        3x3 matrix of the pelvis anatomical axes in the pelvis sensor's frame. Columns correspond to pelvis X, Y, and Z.
    thigh_AF : numpy.ndarray
        3x3 matrix of the thigh anatomical axes in the thigh sensor's frame. Columns correspond to thigh X, Y, and Z.
    R : numpy.ndarray
        Nx3x3 array of rotation matrices from the thigh sensor frame to the pelvis sensor frame.
    zero_angles : bool, optional
        Remove any offset from zero at the start of the angles. Default is False.

    Returns
    -------
    angles : numpy.ndarray
        Nx3 array of hip angles in degrees, with the first column being Flexion - Extension,
        second column being Ad / Abduction, and the third column being Internal - External Rotation.
    """
    # get pelvis anatomical frame
    X = pelvis_AF[:, 0]
    Y = pelvis_AF[:, 1]
    Z = pelvis_AF[:, 2]

    # get the thigh anatomical frame and rotate into pelvis frame
    x = R @ thigh_AF[:, 0]
    y = R @ thigh_AF[:, 1]
    z = R @ thigh_AF[:, 2]

    # form the joint axes
    e1 = Z.copy()
    e3 = y.copy()

    e2 = cross(e3, e1)
    e2 /= norm(e2, axis=1, keepdims=True)

    # compute angles
    sgn = sum(cross(X, e2) * Z, axis=1)
    sgn /= nabs(sgn)
    fe = atan2(sgn * norm(cross(X, e2), axis=1), sum(X * e2, axis=1))

    if 'right' in hip.name.lower():
        sgn = sum(cross(x, e2) * -y, axis=1)
    elif 'left' in hip.name.lower():
        sgn = sum(cross(x, e2) * y, axis=1)
    sgn /= nabs(sgn)
    ier = atan2(sgn * norm(cross(x, e2), axis=1), sum(x * e2, axis=1))

    if 'right' in hip.name.lower():
        aa = -pi / 2 + arccos(sum(e1 * e3, axis=1))
    elif 'left' in hip.name.lower():
        aa = pi / 2 - arccos(sum(e1 * e3, axis=1))
    else:
        raise ValueError('Joint name must have a side in it for angle sign determination.')

    angles = stack((fe, aa, ier), axis=1) * 180 / pi
    if zero_angles:
        angles -= angles[0, :]

    return angles
