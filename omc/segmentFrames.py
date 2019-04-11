"""
Creation of anatomical segment frames from optical motion capture data

GNU GPL v3.0
Lukas Adamowicz
V0.1 - April 10, 2019
"""
from numpy import cross, stack
from numpy.linalg import norm

from pymotion.omc import utility


def pelvis(marker_data, use_cluster=False, R_s_c=None, marker_names='default'):
    """
    Create the pelvis anatomical frame.

    Parameters
    ----------
    marker_data : dictionary
        Dictionary of marker position data. Keys correspond to marker names
    use_cluster : bool, optional
        Use the cluster to segment rotation to compute the anatomical frame. Default is False.
    R_s_c : {None, numpy.ndarray}, optional
        If use_cluster is False, R_s_c is ignored. If use_cluster is True, then a 3x3 rotation matrix must be provided
        that is the rotation from the segment to cluster frame for the pelvis.
    marker_names : {'default', pymotion.omc.utility.MarkerNames}, optional
        Either 'default', which will use the default marker names, or a modified MarkerNames object, with marker names
        as used in the keys of pelvis_data, left_thigh_data, and right_thigh_data.

    Returns
    -------
    pelvis_af : numpy.ndarray
        3x3 matrix representing the pelvis anatomical frame. Also is the rotation from world frame into the pelvis
        anatomical frame. Columns are the x, y, z axes of the anatomical frame.
    """
    # get marker names
    if marker_names == 'default':
        names = utility.MarkerNames
    else:
        names = marker_names

    if use_cluster:
        # first compute the cluster orientation
        R_w_c = utility.create_cluster_frame(marker_data, 'pelvis', names)

        # compute the anatomical frame matrix, which is the world to segment rotation matrix
        pelvis_af = R_s_c.T @ R_w_c

    else:
        # compute the pelvis origin
        origin = utility.compute_pelvis_origin(marker_data[names.left_asis], marker_data[names.right_asis])

        z = marker_data[names.right_asis] - marker_data[names.left_asis]
        z /= norm(z) if z.ndim == 1 else norm(z, axis=1, keepdims=True)

        mid_psis = (marker_data[names.right_psis] + marker_data[names.left_psis]) / 2

        x_tmp = origin - mid_psis

        y = cross(z, x_tmp)
        y /= norm(y) if y.ndim == 1 else norm(y, axis=1, keepdims=True)

        x = cross(y, z)
        x /= norm(x) if x.ndim == 1 else norm(x, axis=1, keepdims=True)

        # create a matrix with columns as the x, y, z, axes of the anatomical frame
        pelvis_af = stack((x, y, z), axis=1) if x.ndim == 1 else stack((x, y, z), axis=2)

    return pelvis_af
