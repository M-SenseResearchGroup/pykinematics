"""
Utility methods for OMC joint angle estimation

GNU GPL v3.0
Lukas Adamowicz
V0.1 - April 10, 2019
"""


def compute_pelvis_origin(left_asis, right_asis):
    """
    Compute the origin of the pelvis.

    Parameters
    ----------
    left_asis : numpy.ndarray
        1 or 2D (Nx3) array of positions of the left asis marker.
    right_asis : numpy.ndarray
        1 or 2D (Nx3) array of positions of the right asis marker.

    Returns
    -------
    origin : numpy.ndarray
        Array of the position of the pelvis origin.

    References
    ----------
    Ren et al.  "Whole body inverse dynamics over a complete gait cycle based only on measured kinematics."
    J. of Biomech. 2008
    """
    origin = (left_asis + right_asis) / 2

    return origin


def compute_thigh_origin(lat_fem_ep, med_fem_ep):
    """
    Compute the origin of the pelvis.

    Parameters
    ----------
    lat_fem_ep : numpy.ndarray
        1 or 2D (Nx3) array of positions of the lateral femoral epicondyle marker.
    med_fem_ep : numpy.ndarray
        1 or 2D (Nx3) array of positions of the medial femoral epicondyle marker.

    Returns
    -------
    origin : numpy.ndarray
        Array of the position of the thigh origin

    References
    ----------
    Ren et al.  "Whole body inverse dynamics over a complete gait cycle based only on measured kinematics."
    J. of Biomech. 2008
    """
    origin = (lat_fem_ep + med_fem_ep) / 2

    return origin


def compute_shank_origin(lat_mall, med_mall):
    """
    Compute the origin of the pelvis.

    Parameters
    ----------
    lat_mall : numpy.ndarray
        1 or 2D (Nx3) array of positions of the lateral malleolus marker.
    med_mall : numpy.ndarray
        1 or 2D (Nx3) array of positions of the medial malleolus marker.

    Returns
    -------
    origin : numpy.ndarray
        Array of the position of the pelvis origin.

    References
    ----------
    Ren et al.  "Whole body inverse dynamics over a complete gait cycle based only on measured kinematics."
    J. of Biomech. 2008
    """
    origin = (lat_mall + med_mall) / 2

    return origin
