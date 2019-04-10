from . import utility


def process_static(marker_data, marker_names, window):
    """
    Process data from a static calibration trial to create the anatomical frames and create constant cluster to
    anatomical frame rotations.

    Parameters
    ----------

    Returns
    -------

    """
    # find the most still period in the data, and the index of that window
    still_data, still_ind = utility.find_most_still(marker_data, window, return_index=True)

    """
    # aggregate all the marker data for the pelvis, thigh, and shank segments
    data = tuple()
    data += tuple(body.pelvis.mocap.mkrs[marker][trial] for marker in body.pelvis.mocap.mkr_names)
    data += tuple(body.l_thigh.mocap.mkrs[marker][trial] for marker in body.l_thigh.mocap.mkr_names)
    data += tuple(body.r_thigh.mocap.mkrs[marker][trial] for marker in body.r_thigh.mocap.mkr_names)
    data += tuple(body.l_shank.mocap.mkrs[marker][trial] for marker in body.l_shank.mocap.mkr_names)
    data += tuple(body.r_shank.mocap.mkrs[marker][trial] for marker in body.r_shank.mocap.mkr_names)

    # find the most still period in the data, and the index of that window
    still_data, still_ind = U.find_most_still(data, int(window * body.pelvis.mocap.fs[trial]), return_index=True)

    # get a list of all the marker names in the same order as they appear in the still data
    mkr_names = tuple()
    mkr_names += tuple(marker for marker in body.pelvis.mocap.mkr_names)
    mkr_names += tuple(marker for marker in body.l_thigh.mocap.mkr_names)
    mkr_names += tuple(marker for marker in body.r_thigh.mocap.mkr_names)
    mkr_names += tuple(marker for marker in body.l_shank.mocap.mkr_names)
    mkr_names += tuple(marker for marker in body.r_shank.mocap.mkr_names)

    # associate names with data
    markers = dict()
    for data, name in zip(still_data, mkr_names):
        markers[name] = data

    # -----------------------------------------------
    #              PELVIS
    # -----------------------------------------------
    compute_segment_origin(body.pelvis, calibration=True, markers=(markers['left_asis'], markers['right_asis']))

    z = markers['right_asis'] - markers['left_asis']
    z /= norm(z)

    mid_psis = (markers['right_psis'] + markers['left_psis']) / 2
    x_tmp = body.pelvis.mocap.origin['Calibration'] - mid_psis

    y = cross(z, x_tmp)
    y /= norm(y)

    x = cross(y, z)
    x /= norm(x)
    body.pelvis.mocap.R['Calibration'] = stack((x, y, z), axis=1)
    x, x_tmp, y, z = None, None, None, None

    if hip_joint_trial is not None:
        compute_segment_origin(body.pelvis, hip_joint_trial, calibration=False)
        compute_segment_orientation(body.pelvis, hip_joint_trial, pelvis_markers=True)

        bias_comp_least_squares_cor(body.l_thigh, hip_joint_trial, body.pelvis, side='left')
        bias_comp_least_squares_cor(body.r_thigh, hip_joint_trial, body.pelvis, side='right')

    if pelvis_z == 'joints':
        # transform joint centers into world frame
        left_jc = ((body.pelvis.mocap.R['Calibration'] @ body.pelvis.mocap.jc['Left Hip']).reshape((-1, 3))
                   + body.pelvis.mocap.origin['Calibration']).flatten()
        right_jc = ((body.pelvis.mocap.R['Calibration'] @ body.pelvis.mocap.jc['Right Hip']).reshape((-1, 3))
                    + body.pelvis.mocap.origin['Calibration']).flatten()

        # recompute the anatomical axis
        z = right_jc - left_jc
        z /= norm(z)

        mid_psis = (markers['right_psis'] + markers['left_psis']) / 2
        x_tmp = body.pelvis.mocap.origin['Calibration'] - mid_psis

        y = cross(z, x_tmp)
        y /= norm(y)

        x = cross(y, z)
        x /= norm(x)
        body.pelvis.mocap.R['Calibration'] = stack((x, y, z), axis=1)
        x, x_tmp, y, z = None, None, None, None

        # transform the joint coordinates back into the new anatomical frame
        body.pelvis.mocap.jc['Left Hip'] = body.pelvis.mocap.R['Calibration'].T \
                                           @ (left_jc - body.pelvis.mocap.origin['Calibration'])

        body.pelvis.mocap.jc['Right Hip'] = body.pelvis.mocap.R['Calibration'].T \
                                            @ (right_jc - body.pelvis.mocap.origin['Calibration'])

    # compute cluster rotation
    xc = markers['sacrum_cluster3'] - markers['sacrum_cluster2']
    xc /= norm(xc)

    yc_tmp = markers['sacrum_cluster1'] - markers['sacrum_cluster2']

    zc = cross(xc, yc_tmp)
    zc /= norm(zc)

    yc = cross(zc, xc)
    yc /= norm(yc)

    body.pelvis.mocap.R_cw['Calibration'] = stack((xc, yc, zc), axis=1)
    xc, yc, yc_tmp, zc = None, None, None, None

    # compute static segment to cluster rotation
    body.pelvis.mocap.R_sc = body.pelvis.mocap.R_cw['Calibration'].T @ body.pelvis.mocap.R['Calibration']

    # -----------------------------------------------
    #              LEFT THIGH
    # -----------------------------------------------
    if hip_joint_trial is not None:
        hjc = ((body.pelvis.mocap.R['Calibration'] @ body.pelvis.mocap.jc['Left Hip']).reshape((-1, 3))
               + body.pelvis.mocap.origin['Calibration']).flatten()

        # wind = int(window * body.pelvis.mocap.fs[hip_joint_trial])
        # pad = int(ceil(wind / 2))
        # hjc_mov_avg, _, pad = U.mov_avg(hjc_time, int(window * body.pelvis.mocap.fs[hip_joint_trial]))

        # hjc = mean(hjc_time[still_ind - pad:still_ind + pad + 1, :], axis=0)

    mid_ep = (markers['left_lat_femoral_epicondyle'] + markers['left_med_femoral_epicondyle']) / 2

    y = hjc - mid_ep
    y /= norm(y)

    z_tmp = (markers['left_med_femoral_epicondyle'] - markers['left_lat_femoral_epicondyle'])

    x = cross(y, z_tmp)
    x /= norm(x)

    z = cross(x, y)
    z /= norm(z)

    body.l_thigh.mocap.R['Calibration'] = stack((x, y, z), axis=1)
    x, y, z, z_tmp = None, None, None, None  # clear so that they don't get used by another segment

    # compute cluster orientation
    xc = markers['left_thigh_cluster3'] - markers['left_thigh_cluster2']
    xc /= norm(xc)

    yc_tmp = markers['left_thigh_cluster1'] - markers['left_thigh_cluster2']

    zc = cross(xc, yc_tmp)
    zc /= norm(zc)

    yc = cross(zc, xc)
    yc /= norm(yc)

    body.l_thigh.mocap.R_cw['Calibration'] = stack((xc, yc, zc), axis=1)
    xc, yc, yc_tmp, zc = None, None, None, None

    # compute static segment to cluster orientation
    body.l_thigh.mocap.R_sc = body.l_thigh.mocap.R_cw['Calibration'].T @ body.l_thigh.mocap.R['Calibration']

    # -----------------------------------------------
    #              Right THIGH
    # -----------------------------------------------
    if hip_joint_trial is not None:
        hjc = ((body.pelvis.mocap.R['Calibration'] @ body.pelvis.mocap.jc['Right Hip']).reshape((-1, 3))
               + body.pelvis.mocap.origin['Calibration']).flatten()

        # hjc_mov_avg, _, pad = U.mov_avg(hjc_time, int(window * body.pelvis.mocap.fs[hip_joint_trial]))
        # wind = int(window * body.pelvis.mocap.fs[hip_joint_trial])
        # pad = int(ceil(wind / 2))

        # hjc = hjc_mov_avg[still_ind + pad]
        # hjc = mean(hjc_time[still_ind - pad:still_ind + pad + 1, :], axis=0)

    mid_ep = (markers['right_med_femoral_epicondyle'] + markers['right_lat_femoral_epicondyle']) / 2

    y = hjc - mid_ep
    y /= norm(y)

    z_tmp = markers['right_lat_femoral_epicondyle'] - markers['right_med_femoral_epicondyle']

    x = cross(y, z_tmp)
    x /= norm(x)

    z = cross(x, y)
    z /= norm(z)

    body.r_thigh.mocap.R['Calibration'] = stack((x, y, z), axis=1)
    x, y, z, z_tmp = None, None, None, None

    # cluster orientation
    xc = markers['right_thigh_cluster3'] - markers['right_thigh_cluster2']
    xc /= norm(xc)

    yc_tmp = markers['right_thigh_cluster1'] - markers['right_thigh_cluster2']

    zc = cross(xc, yc_tmp)
    zc /= norm(zc)

    yc = cross(zc, xc)
    yc /= norm(yc)

    body.r_thigh.mocap.R_cw['Calibration'] = stack((xc, yc, zc), axis=1)
    xc, yc, yc_tmp, zc = None, None, None, None

    # static segment to cluster
    body.r_thigh.mocap.R_sc = body.r_thigh.mocap.R_cw['Calibration'].T @ body.r_thigh.mocap.R['Calibration']
    """


