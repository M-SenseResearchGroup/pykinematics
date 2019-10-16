"""
Utility methods for OMC joint angle estimation

GNU GPL v3.0
Lukas Adamowicz
V0.1 - April 10, 2019
"""
from numpy import cross, stack
from numpy.linalg import norm
from dataclasses import dataclass


@dataclass
class MarkerNames:
    # pelvis markers
    left_asis: str = 'left_asis'
    right_asis: str = 'right_asis'
    left_psis: str = 'left_psis'
    right_psis: str = 'right_psis'

    # pelvis cluster markers
    pelvis_c1: str = 'sacrum_cluster1'
    pelvis_c2: str = 'sacrum_cluster2'
    pelvis_c3: str = 'sacrum_cluster3'

    # left thigh markers
    left_lep: str = 'left_lat_femoral_epicondyle'
    left_mep: str = 'left_med_femoral_epicondyle'

    # left thigh cluster markers
    left_thigh_c1: str = 'left_thigh_cluster1'
    left_thigh_c2: str = 'left_thigh_cluster2'
    left_thigh_c3: str = 'left_thigh_cluster3'

    # left thigh markers
    right_lep: str = 'right_lat_femoral_epicondyle'
    right_mep: str = 'right_med_femoral_epicondyle'

    # right thigh cluster markers
    right_thigh_c1: str = 'right_thigh_cluster1'
    right_thigh_c2: str = 'right_thigh_cluster2'
    right_thigh_c3: str = 'right_thigh_cluster3'


def create_cluster_frame(marker_data, segment_name, marker_names='default'):
    """
    Create a cluster reference frame for the specified segment.

    Parameters
    ----------
    marker_data : dict
        Dictionary of marker data, where keys are the marker names.
    segment_name : {'pelvis', 'left_thigh', 'right_thigh', 'left_shank', 'right_shank'}
        Name of the segment to calculate the cluster for. Must match a prefix of a cluster name in marker_names
    marker_names : {'default', MarkerNames}, optional
        Either 'default' which will use the default marker names (see MarkerNames class), or a modified MarkerNames
        object where the names are changed to match those in the marker_data keys.

    Returns
    -------
    R_w_c : numpy.ndarray
        Either a 3x3 or Nx3x3 array of rotation matrices of the rotation from the world frame to the cluster frame.
        The first, second, and third columns are also the cluster frame x, y, and z axes respectively.
    """
    # get the marker names to use
    if marker_names == 'default':
        names = MarkerNames()
    else:
        names = marker_names

    # create the cluster names
    name_c1 = segment_name + '_c1'
    name_c2 = segment_name + '_c2'
    name_c3 = segment_name + '_c3'

    # create the cluster frame
    x = marker_data[names.__dict__[name_c3]] - marker_data[names.__dict__[name_c2]]
    x /= norm(x) if x.ndim == 1 else norm(x, axis=1, keepdims=True)

    y_tmp = marker_data[names.__dict__[name_c1]] - marker_data[names.__dict__[name_c2]]

    z = cross(x, y_tmp)
    z /= norm(z) if z.ndim == 1 else norm(z, axis=1, keepdims=True)

    y = cross(z, x)
    y /= norm(y) if y.ndim == 1 else norm(y, axis=1, keepdims=True)

    # create and return the matrix of axes
    R_w_c = stack((x, y, z), axis=1) if x.ndim == 1 else stack((x, y, z), axis=2)
    return R_w_c


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
