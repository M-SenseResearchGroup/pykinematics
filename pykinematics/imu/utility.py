"""
General utility methods used in the calculation of joint angles from inertial sensors

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import array, zeros, argsort, dot, arccos, cross, cos, sin, real
from numpy.linalg import norm, eig


def calc_derivative(x, dt, order=2):
    """
    Calculate the 2nd or 4th order derivative of a sequence.

    Parameters
    ----------
    x : numpy.ndarray
        1 or 2D array to take the derivative of.
    dt : float
        Time difference for all points.  This cannot handle varying time differences.
    order : {4, 2}, optional
        Order of the derivative to calculate.  Default is 2nd order.

    Returns
    -------
    dx : numpy.ndarray
        Derivative of x.
    """
    dx = zeros(x.shape)

    if order == 4:
        dx[2:-2] = (x[:-4] - 8 * x[1:-3] + 8 * x[3:-1] + x[4:]) / (12 * dt)

        # edges
        dx[:2] = (-25 * x[:2] + 48 * x[1:3] - 36 * x[2:4] + 16 * x[3:5] - 3 * x[4:6]) / (12 * dt)
        dx[-2:] = (25 * x[-2:] - 48 * x[-3:-1] + 36 * x[-4:-2] - 16 * x[-5:-3] + 3 * x[-6:-4]) / (12 * dt)
    elif order == 2:
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)

        # edges
        dx[0] = (-3 * x[0] + 4 * x[1] - x[2]) / (2 * dt)
        dx[-1] = (3 * x[-1] - 4 * x[-2] + x[-3]) / (2 * dt)

    return dx


def quat_mult(q1, q2):
    """
    Multiply quaternions

    Parameters
    ----------
    q1 : numpy.ndarray
        1x4 array representing a quaternion
    q2 : numpy.ndarray
        1x4 array representing a quaternion

    Returns
    -------
    q : numpy.ndarray
        1x4 quaternion product of q1*q2
    """
    if q1.shape != (1, 4) and q1.shape != (4, 1) and q1.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q1 has more or less than 4 elements')
    if q2.shape != (1, 4) and q2.shape != (4, 1) and q2.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q2 has more or less than 4 elements')
    if q1.shape == (4, 1):
        q1 = q1.T

    Q = array([[q2[0], q2[1], q2[2], q2[3]],
               [-q2[1], q2[0], -q2[3], q2[2]],
               [-q2[2], q2[3], q2[0], -q2[1]],
               [-q2[3], -q2[2], q2[1], q2[0]]])

    return q1 @ Q


def quat_conj(q):
    """
    Compute the conjugate of a quaternion

    Parameters
    ----------
    q : numpy.ndarray
        Nx4 array of N quaternions to compute the conjugate of.

    Returns
    -------
    q_conj : numpy.ndarray
        Nx4 array of N quaternion conjugats of q.
    """
    return q * array([1, -1, -1, -1])


def quat_inv(q):
    """
    Invert a quaternion

    Parameters
    ----------
    q : numpy.ndarray
        1x4 array representing a quaternion

    Returns
    -------
    q_inv : numpy.ndarray
        1x4 array representing the inverse of q
    """
    q_conj = q * array([1, -1, -1, -1])
    return q_conj / sum(q ** 2)


def quat2matrix(q):
    """
    Transform quaternion to rotation matrix

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion

    Returns
    -------
    R : numpy.ndarray
        Rotation matrix
    """
    if q.ndim == 1:
        s = norm(q)
        R = array([[1 - 2 * s * (q[2] ** 2 + q[3] ** 2), 2 * s * (q[1] * q[2] - q[3] * q[0]),
                    2 * s * (q[1] * q[3] + q[2] * q[0])],
                   [2 * s * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * s * (q[1] ** 2 + q[3] ** 2),
                    2 * s * (q[2] * q[3] - q[1] * q[0])],
                   [2 * s * (q[1] * q[3] - q[2] * q[0]), 2 * s * (q[2] * q[3] + q[1] * q[0]),
                    1 - 2 * s * (q[1] ** 2 + q[2] ** 2)]])
    elif q.ndim == 2:
        s = norm(q, axis=1)
        R = array([[1 - 2 * s * (q[:, 2]**2 + q[:, 3]**2), 2 * s * (q[:, 1] * q[:, 2] - q[:, 3] * q[:, 0]),
                    2 * s * (q[:, 1] * q[:, 3] + q[:, 2] * q[:, 0])],
                   [2 * s * (q[:, 1] * q[:, 2] + q[:, 3] * q[:, 0]), 1 - 2 * s * (q[:, 1]**2 + q[:, 3]**2),
                    2 * s * (q[:, 2] * q[:, 3] - q[:, 1] * q[:, 0])],
                   [2 * s * (q[:, 1] * q[:, 3] - q[:, 2] * q[:, 0]), 2 * s * (q[:, 2] * q[:, 3] + q[:, 1] * q[:, 0]),
                    1 - 2 * s * (q[:, 1]**2 + q[:, 2]**2)]])
        R = R.transpose([2, 0, 1])
    return R


def quat_mean(q):
    """
    Calculate the mean of an array of quaternions

    Parameters
    ----------
    q : numpy.ndarray
        Nx4 array of quaternions

    Returns
    -------
    q_mean : numpy.array
        Mean quaternion
    """
    M = q.T @ q

    vals, vecs = eig(M)  # Eigenvalues and vectors of M
    sort_ind = argsort(vals)  # get the indices of the eigenvalues sorted

    q_mean = real(vecs[:, sort_ind[-1]])

    # ensure no discontinuities
    if q_mean[0] < 0:
        q_mean *= -1

    return q_mean


def vec2quat(v1, v2):
    """
    Find the rotation quaternion between two vectors. Rotate v1 onto v2
    Parameters
    ----------
    v1 : numpy.ndarray
        Vector 1
    v2 : numpy.ndarray
        Vector 2

    Returns
    -------
    q : numpy.ndarray
        Quaternion representing the rotation from v1 to v2
    """
    angle = arccos(dot(v1.flatten(), v2.flatten()) / (norm(v1) * norm(v2)))

    # Rotation axis is always normal to two vectors
    axis = cross(v1.flatten(), v2.flatten())
    axis = axis / norm(axis)  # normalize

    q = zeros(4)
    q[0] = cos(angle / 2)
    q[1:] = axis * sin(angle / 2)
    q /= norm(q)

    return q
