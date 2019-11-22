"""
Methods for calculating OMC joint angles

GNU GPL v3.0
Lukas Adamowicz
V0.1 - April 10, 2019
"""
from numpy import cross, sum as nsum, arctan2 as atan2, arccos, abs as nabs, pi, stack
from numpy.linalg import norm


def hip(pelvis_frames, thigh_frames, side):
    """
    Compute hip angles.

    Parameters
    ----------
    pelvis_frames : numpy.ndarray
        Nx3x3 array of anatomical frames of the pelvis in a world frame. N time samples of 3x3 matrices, of which the
        first column is the pelvis X-axis, second column is the pelvis Y-axis, and third column is the pelvis Z-axis
    thigh_frames: numpy.ndarray
        Nx3x3 array of anatomical frames of the thigh for N time points in the world frame. Each 3x3 matrix is comprised
        of columns of thigh x-axis, y-axis, and z-axis, in that order.
    side : {'left', 'right'}
        Side the angles are being computed for. Used for sign determination of angles.

    Returns
    -------
    hip_angles : numpy.ndarray
        Nx3 array of hip angles, with the first column being flexion-extension, second column being ad/abduction,
        and third column being internal-external rotation.

    References
    ----------
    Wu et al. "ISB recommendations on definitions of joint coordinate systems of various joints for the reporting of
    human joint motion - part I: ankle, hip, and spine." J. of Biomech. Vol. 35. 2002.
    Dabirrahmani et al. "Modification of the Grood and Suntay Joint Coordinate System equations for knee joint flexion."
    Med. Eng. and Phys. Vol. 39. 2017.
    Grood et al. "A joint coordinate system for the clinical description of three-dimensional motions: application to
    the knee." J. of Biomech. Engr. Vol. 105. 1983.
    """
    # extract the proximal (pelvis) segment axes
    X = pelvis_frames[:, :, 0]
    Z = pelvis_frames[:, :, 2]

    # extract the distal (thigh) segment axes
    x = thigh_frames[:, :, 0]
    y = thigh_frames[:, :, 1]

    # create the hip joint axes
    e1 = Z.copy()
    e3 = y.copy()

    e2 = cross(e3, e1)
    e2 /= norm(e2, axis=1, keepdims=True)

    # compute the angles by finding the angle between specific axes
    sgn = nsum(cross(X, e2) * Z, axis=1)
    sgn /= nabs(sgn)
    fe = atan2(sgn * norm(cross(X, e2), axis=1), nsum(X * e2, axis=1))

    if side.lower() == 'right':
        sgn = nsum(cross(x, e2) * -y, axis=1)
    elif side.lower() == 'left':
        sgn = nsum(cross(x, e2) * y, axis=1)
    else:
        raise ValueError('Side must be "left" or "right".')

    sgn /= nabs(sgn)
    ier = atan2(sgn * norm(cross(x, e2), axis=1), nsum(x * e2, axis=1))

    if side.lower() == 'right':
        aa = -pi / 2 + arccos(nsum(e1 * e3, axis=1))
    elif side.lower() == 'left':
        aa = pi / 2 - arccos(nsum(e1 * e3, axis=1))

    return stack((fe, aa, ier), axis=1) * 180 / pi