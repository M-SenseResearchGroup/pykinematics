"""
Methods for calibration of OMC data.

GNU GPL v3.0
Lukas Adamowicz
V0.1 - April 10, 2019
"""
from numpy import cross, stack, zeros, isnan, sum as nsum, array, mean
from numpy.linalg import norm, solve

from pykinematics.omc import utility
from pykinematics.omc import segmentFrames
from pykinematics import common


def compute_hip_center(pelvis_data, thigh_data, R, origin, marker_names='default', tol=1e-4):
    """
    Compute the hip joint center location using a bias compensated least squares estimation.

    Parameters
    ----------
    pelvis_data : dict
        Dictionary of all the marker data for the pelvis. Each key is a marker name.
    thigh_data : dict
        Dictionary of all the marker data for the thigh. Each key is a marker name
    R : numpy.ndarray
        Rotation matrix or array of rotation matrices representing the rotation from world to local reference frame.
    origin : numpy.ndarray
        Vector of the origin position to use when rotating and transforming the marker positions into a local
        reference frame.
    marker_names : {'default', MarkerNames}, optional
        Either 'default' to use the default marker names, or a MarkerNames object with the names used in pelvis_data
        and thigh_data specified.
    tol : float, optional
        Tolerance for the bias compensation of the joint center. Default is 1e-4.

    Returns
    -------
    c : numpy.ndarray
        Vector from the provided origin to the joint center, expressed in the local reference frame.

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

    # create a mask for NaN values
    mask = ~isnan(R).any(axis=2).any(axis=1)  # check the world to local frame rotation
    mask &= ~isnan(pelvis_data[names.pelvis_c2]).any(axis=1)  # check the thigh cluster origin

    # check all the marker data for nan as all the data has to be the same length
    for mkr in thigh_data.keys():
        mask &= ~isnan(thigh_data[mkr]).any(axis=1)

    # translate and rotate all the thigh marker coordinates into the cluster reference frame
    thigh_rot = dict()
    for mkr in thigh_data.keys():
        thigh_rot[mkr] = (R[mask] @ (thigh_data[mkr][mask]
                                     - origin[mask]).reshape((-1, 3, 1))).reshape((-1, 3))

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
            u = thigh_rot[mk].reshape((-1, 3, 1)) - c
            u2bar = 1 / u.shape[0] * nsum(u.transpose([0, 2, 1]) @ u, axis=0)
            sigma2[i] = 1 / (4 * u2bar * u.shape[0]) * nsum((nsum(u * u, axis=1) - u2bar) ** 2)

        sigma2avg = mean(sigma2)
        delta_b = zeros((3, 1))
        for mk in thigh_rot.keys():
            delta_b += 1 / thigh_rot[mk].shape[0] * nsum(thigh_rot[mk].reshape((-1, 3, 1)) - c, axis=0)

        delta_b *= 2 * sigma2avg

        cnew = solve(A, b - delta_b)

        delta_c = (cnew - c) / c
        c = cnew.copy()

    return c


def process_static(static_data, hip_center_data, window,
                   marker_names='default'):
    """
    Process data from a static calibration trial to create the anatomical frames and create constant cluster to
    anatomical frame rotations.

    Parameters
    ----------
    static_data : tuple
        3-tuple of dictionaries. The first is the dictionary of pelvis data, the second is the dictionary of
        left-thigh data, and the third is the dictionary of right-thigh data during a static standing trial.
    hip_center_data : tuple
        3-tuple of dictionaries. The first is the dictionary of pelvis data, the second is the dictionary of
        left-thigh data, and the third is the dictionary of right-thigh data during a trial where the hip
        joint center can be calculated (enough rotation in axes is present to compute the joint center).
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
    # extract the data from the tuples
    pelvis_data, left_thigh_data, right_thigh_data = static_data
    pelvis_jc_data, left_thigh_jc_data, right_thigh_jc_data = hip_center_data

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
    # compute the pelvis origin, will be needed later
    pelvis_o = utility.compute_pelvis_origin(markers[names.left_asis], markers[names.right_asis])

    # create a matrix with columns as the x, y, z, axes of the anatomical frame
    # this is also the rotation matrix from pelvis frame to world frame
    pelvis_af = segmentFrames.pelvis(markers, use_cluster=False, marker_names=names)

    # create the cluster frame
    pelvis_R_c_w = utility.create_cluster_frame(markers, 'pelvis', marker_names=names)

    # create the constant segment to cluster rotation matrix
    pelvis_R_s_c = pelvis_R_c_w.T @ pelvis_af

    # -----------------------------------------------
    #              JOINT CENTERS
    # -----------------------------------------------
    # first compute the anatomical frame using the data provided for joint center computation
    pelvis_jc_af = segmentFrames.pelvis(pelvis_jc_data, use_cluster=False, marker_names=names)

    # compute the origin using the data provided for joint center computation
    pelvis_jc_o = utility.compute_pelvis_origin(pelvis_jc_data[names.left_asis], pelvis_jc_data[names.right_asis])

    # compute the joint center locations in the pelvis frame
    right_jc_p = compute_hip_center(pelvis_jc_data, right_thigh_jc_data, pelvis_jc_af.transpose([0, 2, 1]), pelvis_jc_o,
                                    marker_names=names)
    left_jc_p = compute_hip_center(pelvis_jc_data, left_thigh_jc_data, pelvis_jc_af.transpose([0, 2, 1]), pelvis_jc_o,
                                   marker_names=names)

    # -----------------------------------------------
    #              Right THIGH
    # -----------------------------------------------
    # first transform the left hip joint center coordinates back into the world frame
    right_hjc = pelvis_af @ right_jc_p.flatten() + pelvis_o

    # create the anatomical frame matrix, also the left thigh to world rotation matrix
    rthigh_af = segmentFrames.thigh(markers, 'right', use_cluster=False, hip_joint_center=right_hjc, marker_names=names)

    # compute the cluster orientation, also cluster to world
    rthigh_R_c_w = utility.create_cluster_frame(markers, 'right_thigh', names)

    # compute the constant segment to cluster rotation matrix
    rthigh_R_s_c = rthigh_R_c_w.T @ rthigh_af

    # -----------------------------------------------
    #              LEFT THIGH
    # -----------------------------------------------
    # first transform the left hip joint center coordinates back into the world frame
    left_hjc = pelvis_af @ left_jc_p.flatten() + pelvis_o

    # create the anatomical frame matrix, also the left thigh to world rotation matrix
    lthigh_af = segmentFrames.thigh(markers, 'left', use_cluster=False, hip_joint_center=left_hjc, marker_names=names)

    # compute the cluster orientation, also cluster to world rotation
    lthigh_R_c_w = utility.create_cluster_frame(markers, 'left_thigh', names)

    # compute the constant segment to cluster rotation matrix
    lthigh_R_s_c = lthigh_R_c_w.T @ lthigh_af

    return (pelvis_af, pelvis_R_c_w, pelvis_R_s_c), (lthigh_af, lthigh_R_c_w, lthigh_R_s_c), (rthigh_af, rthigh_R_c_w,
                                                                                              rthigh_R_s_c)


