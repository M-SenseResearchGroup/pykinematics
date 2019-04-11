"""
Methods for calibration of OMC data.

GNU GPL v3.0
Lukas Adamowicz
V0.1 - April 10, 2019
"""
from numpy import cross, stack, zeros, isnan, sum as nsum, array, mean
from numpy.linalg import norm, solve

from pymotion.omc import utility
from pymotion import common


def compute_hip_center(pelvis_data, thigh_data, side, marker_names='default', tol=1e-4):
    """
    Compute the hip joint center location using a bias compensated least squares estimation.

    Parameters
    ----------
    pelvis_data : dict
        Dictionary of all the marker data for the pelvis. Each key is a marker name.
    thigh_data : dict
        Dictionary of all the marker data for the thigh. Each key is a marker name
    side : {'left', 'right'}
        Side for the thigh. Either 'left' or 'right'.
    marker_names : {'default', MarkerNames}, optional
        Either 'default' to use the default marker names, or a MarkerNames object with the names used in pelvis_data
        and thigh_data specified.
    tol : float, optional
        Tolerance for the bias compensation of the joint center. Default is 1e-4.

    Returns
    -------
    c : numpy.ndarray
        Vector from the pelvis cluster origin (cluster marker 2) to the joint center between the pelvis and thigh,
        expressed in the pelvis cluster reference frame.

    References
    ----------
    Halvorsen, Kjartan.  "Bias compensated least squares estimate of the center of rotation." J. of Biomech.
    Vol. 36. 2003.
    Gamage et al. "New least squares solution for estimating the average center of rotation and the axis of rotation."
    J. of Biomech. Vol. 35. 2002
    """
    if marker_names == 'default':
        names = utility.MarkerNames()
    else:
        names = marker_names

    # compute the pelvis cluster orientation matrix
    R_w_c = utility.create_cluster_frame(pelvis_data, 'pelvis', names)

    # get the origin marker name
    if side == 'left':
        o_name = names.left_thigh_c2
    elif side == 'right':
        o_name = names.right_thigh_c2
    else:
        raise ValueError('side must be "left" or "right".')

    # create a mask for NaN values
    mask = ~isnan(R_w_c).any(axis=2).any(axis=1)  # check the world to cluster rotation
    mask &= ~isnan(thigh_data[o_name]).any(axis=1)  # check the thigh cluster origin

    # check all the marker data for nan as all the data has to be the same length
    for mkr in thigh_data.keys():
        mask &= ~isnan(thigh_data[mkr]).any(axis=1)

    # translate and rotate all the thigh marker coordinates into the cluster reference frame
    thigh_rot = dict()
    for mkr in thigh_data.keys():
        thigh_rot[mkr] = (R_w_c[mask] @ (thigh_data[mkr][mask]
                                         - thigh_data[o_name][mask]).reshape((-1, 3, 1))).reshape((-1, 3))

    n = mask.sum()  # number of samples
    m = len(thigh_rot.keys())  # number of markers

    # create the vectors to store the norms
    norm1 = zeros((m, 3, 1))
    norm2 = zeros((m, 1, 1))
    norm3 = zeros((m, 3, 1))

    for i, mk in enumerate(thigh_rot.keys()):
        norm1[i] = thigh_rot[mk].sum(axis=0).reshape((-1, 3, 1))
        norm2[i] = nsum(thigh_rot[mk].reshape((-1, 1, 3)) @ thigh_rot[mk].reshape((-1, 3, 1)), axis=0)
        norm3[i] = nsum(thigh_rot[mk].reshape((-1, 3, 1)) @ thigh_rot[mk].reshape((-1, 1, 3))
                        @ thigh_rot[mk].reshape((-1, 3, 1)), axis=0)
    norm1 /= n
    norm2 /= n
    norm3 /= n

    A = zeros((3, 3))
    b = zeros((3, 1))

    for i, mk in enumerate(thigh_rot.keys()):
        A += nsum(thigh_rot[mk].reshape((-1, 3, 1)) @ thigh_rot[mk].reshape((-1, 1, 3)), axis=0) / n
        A -= norm1[i] @ norm1[i].T

        b += norm3[i] - norm1[i] * norm2[i]

    A *= 2

    # solve for the initial guess of the joint center location in the pelvis cluster reference frame
    c = solve(A, b)

    # BIAS COMPENSATION
    delta_c = array([1, 1, 1])  # change in joint center coordinates
    while (delta_c > tol).any():
        sigma2 = zeros(len(thigh_rot.keys()))
        for i, mk in enumerate(thigh_rot.keys()):
            u = thigh_rot[mk] - c
            u2bar = 1 / u.shape[0] * nsum(u.transpose([0, 2, 1]) @ u, axis=0)
            sigma2[i] = 1 / (4 * u2bar * u.shape[0]) * nsum((nsum(u * u, axis=1) - u2bar) ** 2)

        sigma2avg = mean(sigma2)
        delta_b = zeros((3, 1))
        for mk in thigh_rot.keys():
            delta_b += 1 / thigh_rot[mk].shape[0] * nsum(thigh_rot[mk] - c, axis=0)

        delta_b *= 2 * sigma2avg

        cnew = solve(A, b - delta_b)

        delta_c = (cnew - c) / c
        c = cnew.copy()

    return c


def process_static(pelvis_data, left_thigh_data, right_thigh_data, left_c_pc, right_c_pc, window,
                   marker_names='default'):
    """
    Process data from a static calibration trial to create the anatomical frames and create constant cluster to
    anatomical frame rotations.

    Parameters
    ----------
    pelvis_data : dictionary
        Dictionary of marker data for the pelvis. Each key is the name of the marker.
    left_thigh_data : dictionary
        Dictionary of marker data for the left thigh. Each key is the name of the marker.
    right_thigh_data : dictionary
        Dictionary of marker data for the right thigh. Each key is the name of the marker.
    left_c_pc : numpy.ndarray
        Vector from the pelvis cluster reference frame origin to the left hip joint center, expressed in the
        pelvis cluster reference frame.
    right_c_pc : numpy.ndarray
        Vector from the pelvis cluster reference frame origin to the right hip joint center, expressed in the
        pelvis cluster reference frame.
    window : int
        Number of samples to window over for finding the most still section of data. Suggested is 1 second worth of
        samples.
    marker_names : {'default', pymotion.omc.utility.MarkerNames}, optional
        Either 'default', which will use the default marker names, or a modified MarkerNames object, with marker names
        as used in the keys of pelvis_data, left_thigh_data, and right_thigh_data.

    Returns
    -------
    pelvis : tuple
        3-tuple of pelvis anatomical frame, rotation from world to pelvis cluster frame, and constant rotation from
        segment frame to cluster frame.
    left_thigh : tuple
        3-tuple of left thigh anatomical frame, rotation from world to the left thigh cluster frame, and constant
        rotation from segment frame to cluster frame.
    right_thigh : tuple
        3-tuple of right thigh anatomical frame, rotation from world to the right thigh cluster frame, and constant
        rotation from segment frame to cluster frame.
    """
    # separate the data from the names
    raw_data = tuple(pelvis_data[name] for name in pelvis_data.keys()) \
               + tuple(left_thigh_data[name] for name in left_thigh_data.keys()) \
               + tuple(right_thigh_data[name] for name in right_thigh_data.keys())
    raw_names = list(pelvis_data.keys()) + list(left_thigh_data.keys()) + list(right_thigh_data.keys())

    # find the most still period in the data, and the index of that window
    still_data, still_ind = common.find_most_still(raw_data, window, return_index=True)

    # associate still data with the names
    markers = dict()
    for data, name in zip(still_data, raw_names):
        markers[name] = data

    # set the marker names
    if marker_names == 'default':
        names = utility.MarkerNames()
    else:
        names = marker_names

    # -----------------------------------------------
    #              PELVIS
    # -----------------------------------------------
    pelvis_o = utility.compute_pelvis_origin(markers[names.left_asis], markers[names.right_asis])

    # create the anatomical axes
    pelvis_z = markers[names.right_asis] - markers[names.left_asis]
    pelvis_z / norm(pelvis_z)

    mid_psis = (markers[names.right_psis] + markers[names.left_psis]) / 2

    pelvis_x_tmp = pelvis_o - mid_psis

    pelvis_y = cross(pelvis_z, pelvis_x_tmp)
    pelvis_y /= norm(pelvis_y)

    pelvis_x = cross(pelvis_y, pelvis_z)
    pelvis_x /= norm(pelvis_x)

    # create a matrix with columns as the x, y, z, axes of the anatomical frame
    # this is also the rotation matrix from world to segment frame
    pelvis_af = stack((pelvis_x, pelvis_y, pelvis_z), axis=1)

    # create the cluster frame
    pelvis_R_w_c = utility.create_cluster_frame(pelvis_data, 'pelvis', marker_names=names)

    # create the constant segment to cluster rotation matrix
    pelvis_R_s_c = pelvis_R_w_c @ pelvis_af.T

    # -----------------------------------------------
    #              Right THIGH
    # -----------------------------------------------
    # first transform the left hip joint center coordinates back into the world frame
    right_hjc = (pelvis_R_w_c.T @ (right_c_pc
                                   + right_thigh_data[names.right_thigh_c2]).reshape((-1, 3, 1))).reshape((-1, 3))
    # compute the midpoint of the epicondyles
    mid_ep = (right_thigh_data[names.right_lep] + right_thigh_data[names.right_mep]) / 2

    # create the axes
    rthigh_y = right_hjc - mid_ep
    rthigh_y /= norm(rthigh_y)

    z_tmp = right_thigh_data[names.right_lep] - right_thigh_data[names.right_mep]

    rthigh_x = cross(rthigh_y, z_tmp)
    rthigh_x /= norm(rthigh_x)

    rthigh_z = cross(rthigh_x, rthigh_y)
    rthigh_z /= norm(rthigh_z)

    # create the anatomical frame matrix, also the world to left thigh rotation matrix
    rthigh_af = stack((rthigh_x, rthigh_y, rthigh_z), axis=1)

    # compute the cluster orientation
    rthigh_R_w_c = utility.create_cluster_frame(right_thigh_data, 'right_thigh', names)

    # compute the constant segment to cluster rotation matrix
    rthigh_R_s_c = rthigh_R_w_c @ rthigh_af.T

    # -----------------------------------------------
    #              LEFT THIGH
    # -----------------------------------------------
    # first transform the left hip joint center coordinates back into the world frame
    left_hjc = (pelvis_R_w_c.T @ (left_c_pc
                                  + left_thigh_data[names.left_thigh_c2]).reshape((-1, 3, 1))).reshape((-1, 3))
    # compute the midpoint of the epicondyles
    mid_ep = (left_thigh_data[names.left_lep] + left_thigh_data[names.left_mep]) / 2

    # create the axes
    lthigh_y = left_hjc - mid_ep
    lthigh_y /= norm(lthigh_y)

    z_tmp = left_thigh_data[names.left_mep] - left_thigh_data[names.left_lep]

    lthigh_x = cross(lthigh_y, z_tmp)
    lthigh_x /= norm(lthigh_x)

    lthigh_z = cross(lthigh_x, lthigh_y)
    lthigh_z /= norm(lthigh_z)

    # create the anatomical frame matrix, also the world to left thigh rotation matrix
    lthigh_af = stack((lthigh_x, lthigh_y, lthigh_z), axis=1)

    # compute the cluster orientation
    lthigh_R_w_c = utility.create_cluster_frame(left_thigh_data, 'left_thigh', names)

    # compute the constant segment to cluster rotation matrix
    lthigh_R_s_c = lthigh_R_w_c @ lthigh_af.T

    return (pelvis_af, pelvis_R_w_c, pelvis_R_s_c), (lthigh_af, lthigh_R_w_c, lthigh_R_s_c), (rthigh_af, rthigh_R_w_c,
                                                                                              rthigh_R_s_c)


