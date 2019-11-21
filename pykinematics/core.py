"""
Core functions for computing joint angles from start to finish

Lukas Adamowicz
May 2019
GNU GPL v3.0
"""
from numpy import mean, diff
from scipy.signal import butter, filtfilt
from warnings import warn

from pykinematics import imu
from pykinematics import omc


class MimuAngles:
    def __init__(self, static_window=1.0, gravity_value=9.81, filter_values=None, angular_velocity_derivative_order=2,
                 joint_center_kwargs=None, orientation_kwargs=None, correct_knee=True, knee_axis_kwargs=None,
                 verbose=True):
        """
        Compute angles from MIMU sensors, from initial raw data through joint angles.

        Parameters
        ----------
        static_window : float, optional
            Window size in seconds for which to use in calibration during static standing. Default is 1s.
        gravity_value : float, optional
            Local gravitational acceleration. Default is 9.81m/s^2
        filter_values : {None, dict}, optional
            Filter values for the inertial and magnetic field data. Default is None, which uses the default settings.
            Providing a dictionary with any of the values modified (ie don't need to specify all 4) will changed the
            specific setting. Each entry is a length 2 tuple, containing first filter order, then cutoff frequency (Hz).
            Default settings and keys:
                - 'Acceleration': (2, 15)
                - 'Angular velocity': (2, 15)
                - 'Angular acceleration': (2, 15)
                - 'Magnetic field': (2, 15)
        angular_velocity_derivative_order : {2, 4}, optional
            Order for the calculation of the angular velocity derivative. Default is 2 for 2nd order.
        joint_center_kwargs : {None, dict}, optional
            Optional joint center computation key-word arguments, or None, for using the defaults. See
            :meth:`pykinematics.imu.joints.Center` for the possible arguments. Default is None.
        orientation_kwargs : {None, dict}, optional
            Optional sensor relative orientation key-word arguments, or NOne, for using the defaults. See
            :meth:`pykinematics.imu.orientation.SSRO` for the possible arguments. Default is None.
        correct_knee : bool, optional
            Correct the knee joint center location by shifting it along the rotation axis closer to the sensors. Default
            is True.
        knee_axis_kwargs : {None, dict}, optional
            Optional knee-axis computation key-word arguments. See :meth:`pykinematics.imu.joints.KneeAxis` for the
            possible arguments. Default is None
        verbose : bool, optional
            Print messages regarding the status of the estimation process. Default is True
        """
        self.static_window = static_window
        self.grav_val = gravity_value

        # set the default filter values
        self.set_default_filter_values()
        if filter_values is not None:  # see if any changes to be included
            if isinstance(filter_values, dict):  # check that the format is followed
                for key in filter_values.keys():  # change the default values
                    self.filt_vals[key] = filter_values[key]

        self.wd_ord = angular_velocity_derivative_order
        assert self.wd_ord == 2 or self.wd_ord == 4, 'The angular velocity derivative order must be either 2 or 4.'

        if joint_center_kwargs is not None:
            self.center_kwargs = joint_center_kwargs
        else:
            self.center_kwargs = {}
        if orientation_kwargs is not None:
            self.orient_kwargs = orientation_kwargs
        else:
            self.orient_kwargs = {}
        if knee_axis_kwargs is not None:
            self.knee_axis_kwargs = knee_axis_kwargs
        else:
            self.knee_axis_kwargs = {}

        self.correct_knee = correct_knee

        # calibrate attributes
        self.acc_scales = None
        self.pelvis_axis, self.l_thigh_axis, self.r_thigh_axis = None, None, None
        self.pelvis_AF, self.l_thigh_AF, self.r_thigh_AF = None, None, None

        self.verbose = verbose

    def estimate(self, task_data, return_orientation=False):
        """
        Estimate joint angles from data during a trial of interest.

        Parameters
        ----------
        task_data : dict
            Dictionary of data from a task to compute the hip joint angles for. Required keys/sensors are Lumbar,
            Left and Right thighs. See the *Notes* for *ImuAngles.calibrate*.
        return_orientation : bool, optional
            Return the estimates of the relative orientation. Default is False.

        Returns
        -------
        left_hip_angles : numpy.ndarray
            (N, 3) array of hip angles for the left hip, in the order Flexion - Extension, Ad / Abduction, and
            Internal - External rotation.
        right_hip_angles : numpy.ndarray
            (N, 3) array of hip angles for the right hip, in the order Flexion - Extension, Ad / Abduction, and
            Internal - External rotation.
        R_Lthigh_lumbar : numpy.ndarray, optional
            Rotations from the left thigh sensor to the lumbar sensor for all time points in the provided data.
            Only returned if `return_orientation` is True.
        R_Rthigh_lumbar : numpy.ndarray, optional
            Rotations from the right thigh sensor to the lumbar sensor for all time points in the provided data.
            Only returned if `return_orientation` is True.
        q_Lthigh_lumbar : numpy.ndarray, optional
            Rotation quaternions from the left thigh sensor to the lumbar sensor for all time points in the provided data.
            Only returned if `return_orientation` is True.
        q_Rthigh_lumbar : numpy.ndarray, optional
            Rotation quaternions from the right thigh sensor to the lumbar sensor for all time points in the provided data.
            Only returned if `return_orientation` is True.
        """
        # check to ensure that the data provided has the required sensors
        MimuAngles._check_required_sensors(task_data, 'trial')

        if self.verbose:
            print('-------------------------------------------------\nPreprocessing trial data...')

        # scale the acceleration data
        for sensor in task_data.keys():
            task_data[sensor]['Acceleration'] *= self.acc_scales[sensor]

        # filter the data
        self._apply_filter_dict(task_data, comp_angular_accel=False)  # don't need angular acceleration

        # compute the relative orientation
        if self.verbose:
            print('Computing sensor relative orientation...')
        srof = imu.orientation.SSRO(grav=self.grav_val, **self.orient_kwargs)
        q_lt_lb, R_lt_lb = MimuAngles._compute_orientation(srof, task_data['Lumbar'], task_data['Left thigh'])
        q_rt_lb, R_rt_lb = MimuAngles._compute_orientation(srof, task_data['Lumbar'], task_data['Right thigh'])

        # compute joint angles
        if self.verbose:
            print('Computing left and right hip joint angles...')
        l_hip_ang = imu.angles.hip_from_frames(self.pelvis_AF, self.l_thigh_AF, R_lt_lb, side='left')
        r_hip_ang = imu.angles.hip_from_frames(self.pelvis_AF, self.r_thigh_AF, R_rt_lb, side='right')

        if return_orientation:
            return l_hip_ang, r_hip_ang, R_lt_lb, R_rt_lb, q_lt_lb, q_rt_lb
        else:
            return l_hip_ang, r_hip_ang

    def calibrate(self, static_data, joint_center_data):
        """
        Calibration by computing the sensor-to-segment alignment.

        Parameters
        ----------
        static_data : dict
            Dictionary of sensor data from a static standing trial. Must contain at least Lumbar, Left thigh, and Right
            thigh keys. Left shank and Right shank are suggested in order to provide an acceleration scale to ensure
            that the measured gravity vectors are equivalent between sensors. See *Notes* for the structure of the
            dictionary.
        joint_center_data : dict
            Dictionary of sensor data from a task to be used for joint center computation. As such the task should have
            sufficient rotation about all possible axes (ie all 3 for the hip, 1 for the knee). Must contain Lumbar,
            thigh, and shank keys. See *Notes*.

        Attributes
        ----------
        self.acc_scales : dict
            Dictionary of acceleration scales for each of the sensors in static_data, required to scale acceleration
            magnitude to that of local gravitational acceleration during static standing
        self.pelvis_axis : numpy.ndarray
            Pelvis fixed axis in the Lumbar sensor's reference frame.
        self.l_thigh_axis : numpy.ndarray
            Left thigh fixed axis in the left thigh's reference frame.
        self.r_thigh_axis : numpy.ndarray
            Right thigh fixed axis in the right thigh's reference frame.
        self.pelvis_AF : tuple
            Tuple of the x, y, and z anatomical axes for the pelvis, in the lumbar sensor frame.
        self.l_thigh_AF : tuple
            Tuple of the x, y, and z anatomical axes for the left thigh, in the left thigh's sensor frame.
        self.r_thigh_AF : tuple
            Tuple of the x, y, and z anatomical axes for the right thigh, in the right thigh's sensor frame.

        Notes
        -----
        Data dictionary key-level schematic:

        - Lumbar
            - Time [s]
            - Acceleration [m/s^2]
            - Angular velocity [rad/s]
            - Magnetic field [not specified]
        - Left thigh
            - Time
            - Acceleration
            - Angular velocity
            - Magnetic field
        - Right thigh
            - Time
            - Acceleration
            - Angular velocity
            - Magnetic field
        - (Left shank)
            - Time
            - Acceleration
            - Angular velocity
            - Magnetic field
        - (Right shank)
            - Time
            - Acceleration
            - Angular velocity
            - Magnetic field
        """
        # check to ensure that the data provided has the required sensors
        MimuAngles._check_required_sensors(static_data, 'static')
        MimuAngles._check_required_sensors(joint_center_data, 'joint center')

        if self.verbose:
            print('\n-----------------------------------------------------\n'
                  'Scaling acceleration and pre-processing the raw data')

        # get the acceleration scales
        self.acc_scales = dict()
        for sensor in static_data.keys():
            self.acc_scales[sensor] = imu.calibration.get_acc_scale(static_data[sensor]['Acceleration'],
                                                                    gravity=self.grav_val)

        # scale the available data
        for sensor in self.acc_scales.keys():
            static_data[sensor]['Acceleration'] *= self.acc_scales[sensor]

            if sensor in list(joint_center_data.keys()):
                joint_center_data[sensor]['Acceleration'] *= self.acc_scales[sensor]
        # issue a warning if any sensors are in the joint center data but not in the static data scales
        for sens in [sens for sens in joint_center_data.keys() if sens not in list(self.acc_scales.keys())]:
            warn(f'Sensor ({sens}) in joint center data has not been scaled due to no scale factor available from '
                 f'static data provided. Performance may suffer as a result.')

        # filter the static data
        self._apply_filter_dict(static_data, comp_angular_accel=False)
        # filter the joint center data
        self._apply_filter_dict(joint_center_data, comp_angular_accel=True)  # need angular accel for this one

        if self.verbose:
            print('Computing joint centers...')
        # compute joint centers from the joint center data
        joint_center = imu.joints.Center(g=self.grav_val, **self.center_kwargs)

        # if the center method is "SAC", we need to compute the relative orientation first
        srof = imu.orientation.SSRO(grav=self.grav_val, **self.orient_kwargs)

        _, jcR_lt_lb = MimuAngles._compute_orientation(srof, joint_center_data['Lumbar'],
                                                       joint_center_data['Left thigh'])
        _, jcR_rt_lb = MimuAngles._compute_orientation(srof, joint_center_data['Lumbar'],
                                                       joint_center_data['Right thigh'])
        _, jcR_ls_lt = MimuAngles._compute_orientation(srof, joint_center_data['Left thigh'],
                                                       joint_center_data['Left shank'])
        _, jcR_rs_rt = MimuAngles._compute_orientation(srof, joint_center_data['Right thigh'],
                                                       joint_center_data['Right shank'])

        # compute the joint centers
        hip_l_lb, hip_l_t, hip_l_res = MimuAngles._compute_center(joint_center, joint_center_data['Lumbar'],
                                                                  joint_center_data['Left thigh'], jcR_lt_lb)
        hip_r_lb, hip_r_t, hip_r_res = MimuAngles._compute_center(joint_center, joint_center_data['Lumbar'],
                                                                  joint_center_data['Right thigh'], jcR_rt_lb)
        knee_l_t, knee_l_s, knee_l_res = MimuAngles._compute_center(joint_center, joint_center_data['Left thigh'],
                                                                    joint_center_data['Left shank'], jcR_ls_lt,
                                                                    self.correct_knee, self.knee_axis_kwargs)
        knee_r_t, knee_r_s, knee_r_res = MimuAngles._compute_center(joint_center, joint_center_data['Right thigh'],
                                                                    joint_center_data['Right shank'], jcR_rs_rt,
                                                                    self.correct_knee, self.knee_axis_kwargs)

        if self.verbose:
            print('------------------------------------------------------------')
            print(
                f'Left hip:  Residual: {hip_l_res:0.3f}\nLumbar: ({hip_l_lb[0] * 100:0.2f}, {hip_l_lb[1] * 100:0.2f}, '
                f'{hip_l_lb[2] * 100:0.2f})cm    Left thigh: ({hip_l_t[0] * 100:0.2f}, {hip_l_t[1] * 100:0.2f}, '
                f'{hip_l_t[2] * 100:0.2f})cm')
            print(
                f'Right hip:  Residual: {hip_r_res:0.3f}\nLumbar: ({hip_r_lb[0] * 100:0.2f}, {hip_r_lb[1] * 100:0.2f}, '
                f'{hip_r_lb[2] * 100:0.2f})cm    Right thigh: ({hip_r_t[0] * 100:0.2f}, {hip_r_t[1] * 100:0.2f}, '
                f'{hip_r_t[2] * 100:0.2f})cm')
            print(
                f'Left knee:  Residual: {knee_l_res:0.3f}\nLeft thigh: ({knee_l_t[0] * 100:0.2f}, '
                f'{knee_l_t[1] * 100:0.2f}, {knee_l_t[2] * 100:0.2f})cm    Left shank: ({knee_l_s[0] * 100:0.2f}, '
                f'{knee_l_s[1] * 100:0.2f}, {knee_l_s[2] * 100:0.2f})cm')
            print(
                f'Right knee:  Residual: {knee_r_res:0.3f}\nRight thigh: ({knee_r_t[0] * 100:0.2f}, '
                f'{knee_r_t[1] * 100:0.2f}, {knee_r_t[2] * 100:0.2f})cm    Right shank: ({knee_r_s[0] * 100:0.2f}, '
                f'{knee_r_s[1] * 100:0.2f}, {knee_r_s[2] * 100:0.2f})cm')
            print('------------------------------------------------------------')
            print('Computing fixed axes and creating anatomical reference frames')

        # compute the fixed axes for the thighs and pelvis
        self.pelvis_axis = imu.joints.fixed_axis(hip_l_lb, hip_r_lb, center_to_sensor=True)
        self.l_thigh_axis = imu.joints.fixed_axis(knee_l_t, hip_l_t, center_to_sensor=True)
        self.r_thigh_axis = imu.joints.fixed_axis(knee_r_t, hip_r_t, center_to_sensor=True)

        # compute the relative orientation between sensors during the static data
        q_lt_lb, _ = MimuAngles._compute_orientation(srof, static_data['Lumbar'], static_data['Left thigh'])
        q_rt_lb, _ = MimuAngles._compute_orientation(srof, static_data['Lumbar'], static_data['Right thigh'])

        # process the static calibration
        AF = imu.calibration.static(q_lt_lb, q_rt_lb, self.pelvis_axis, self.l_thigh_axis, self.r_thigh_axis,
                                    static_data['Lumbar']['Angular velocity'],
                                    static_data['Left thigh']['Angular velocity'],
                                    static_data['Right thigh']['Angular velocity'], static_data['Lumbar']['dt'],
                                    self.static_window)
        self.pelvis_AF, self.l_thigh_AF, self.r_thigh_AF = AF

        if self.verbose:
            print('Calibration complete\n')

    @staticmethod
    def _compute_center(jc, prox, dist, R_dist_prox, correct_knee=False, knee_axis_kwargs=None):
        """
        Compute the joint center

        Parameters
        ----------
        jc : pykinematics.imu.joints.Center
            Initialized joint center computation object
        prox : dict
            Dictionary, containing 'Acceleration', 'Angular velocity', and 'Angular acceleration' readings from a
            sensor.
        dist : dict
            Dictionary, containing 'Acceleration', 'Angular velocity', and 'Angular acceleration' readings from a
            sensor.
        R_dist_prox : numpy.ndarray
            (N, 3, 3) array of rotation matrices that align the distal sensor's frame with the proximal sensor's frame.
        correct_knee : bool, optional
            Whether or not to correct the knee joint. Should only be used for knee joint center computation. Default is
            False.
        knee_axis_kwargs : {None, dict}, optional
            Additional keyword arguments to be passed to the knee axis estimation, which is used for knee joint
            center correction. Default is None.

        Returns
        -------
        prox_joint_center : numpy.ndarray
            Vector from the joint center to the proximal sensor.
        dist_joint_center : numpy.ndarray
            Vector from the joint center to the distal sensor.
        residual : float
            Residual value from the optimization process.
        """
        # run the computation
        prox_jc, dist_jc, res = jc.compute(prox['Acceleration'], dist['Acceleration'],
                                           prox['Angular velocity'], dist['Angular velocity'],
                                           prox['Angular acceleration'], dist['Angular acceleration'], R_dist_prox)
        if correct_knee:
            imu.joints.correct_knee(prox['Angular velocity'], dist['Angular velocity'], prox_jc, dist_jc,
                                    R_dist_prox[0], knee_axis_kwargs)
        return prox_jc, dist_jc, res

    @staticmethod
    def _compute_orientation(sro, sensor1, sensor2):
        """
        Run the orientation estimation filter. Rotation is provided from sensor 2 -> sensor 1

        Parameters
        ----------
        sro : pykinematics.imu.orientation.SROFilter
            Object for estimating the sensor relative orientation
        sensor1 : dict
            Dictionary, containing 'Acceleration', 'Angular velocity', and 'Magnetic field' readings from a sensor.
        sensor2 : dict
            Dictionary, containing 'Acceleration', 'Angular velocity', and 'Magnetic field' readings from a sensor,
            which will be used to find the rotation from sensor2's frame to sensor1's frame

        Returns
        -------
        q_21 : numpy.ndarray
            (N, 4) array quaternions representing the rotation to align sensor2's reference frame with that of sensor1's
        R_21 : numpy.ndarray
            (N, 3, 3) array of rotation matrices corresponding to the quaternions of 'q'
        """
        x = sro.run(sensor1['Acceleration'], sensor2['Acceleration'], sensor1['Angular velocity'],
                    sensor2['Angular velocity'], sensor1['Magnetic field'], sensor2['Magnetic field'], sensor1['dt'])
        # x is the length 10 vector of [gravity_1, gravity_2, q_21]
        R = imu.utility.quat2matrix(x[:, 6:])  # convert to a rotation matrix

        return x[:, 6:], R

    def _apply_filter_dict(self, data, comp_angular_accel=False):
        """
        Apply a filter to a whole dictionary of sensor data, and calculate angular acceleration if necessary.

        Parameters
        ----------
        data : dict
            Dictionary of data to apply the filter to
        comp_angular_accel : bool, optional
            Compute and filter angular acceleration. Default is False
        """
        for sensor in data.keys():
            # compute the sampling time difference, 1/f_sample
            data[sensor]['dt'] = mean(diff(data[sensor]['Time']))
            # apply the specified filter to the acceleration
            data[sensor]['Acceleration'] = MimuAngles._apply_filter(data[sensor]['Acceleration'], data[sensor]['dt'],
                                                                    self.filt_vals['Acceleration'][0],
                                                                    self.filt_vals['Acceleration'][1])
            # apply the specified filter to the angular velocity
            data[sensor]['Angular velocity'] = MimuAngles._apply_filter(data[sensor]['Angular velocity'],
                                                                        data[sensor]['dt'],
                                                                        self.filt_vals['Angular velocity'][0],
                                                                        self.filt_vals['Angular velocity'][1])
            # apply the specified filter to the magnetic field reading
            data[sensor]['Magnetic field'] = MimuAngles._apply_filter(data[sensor]['Magnetic field'],
                                                                      data[sensor]['dt'],
                                                                      self.filt_vals['Magnetic field'][0],
                                                                      self.filt_vals['Magnetic field'][1])

            if comp_angular_accel:
                data[sensor]['Angular acceleration'] = imu.utility.calc_derivative(data[sensor]['Angular velocity'],
                                                                                   data[sensor]['dt'],
                                                                                   order=self.wd_ord)
                data[sensor]['Angular acceleration'] = MimuAngles._apply_filter(data[sensor]['Angular acceleration'],
                                                                                data[sensor]['dt'],
                                                                                self.filt_vals['Angular acceleration'][0],
                                                                                self.filt_vals['Angular acceleration'][1])

    @staticmethod
    def _apply_filter(x, dt, filt_order, filt_cutoff):
        """
        Apply a filter to the data

        Parameters
        ----------
        x : numpy.ndarray
            (N, M) array of data to be filtered. Will filter along the 0th axis (N)
        dt : float
            Sampling time difference between samples.
        filt_order : int
            Order of the filter to be applied
        filt_cutoff : float
            Cutoff frequency of the filter to be applied
        """
        fc = butter(filt_order, filt_cutoff * 2 * dt)
        if x.ndim == 2:
            x = filtfilt(fc[0], fc[1], x, axis=0)
        elif x.ndim == 1:
            x = filtfilt(fc[0], fc[1], x)

        return x

    def set_default_filter_values(self):
        """
        Set the filter values to the default:

        Angular velocity : (2, 15)
        Acceleration : (2, 15)
        Angular acceleration : (2, 15)
        Magnetic field : (2, 15)
        """
        self.filt_vals = {'Angular velocity': (2, 15), 'Acceleration': (2, 15), 'Angular acceleration': (2, 15),
                          'Magnetic field': (2, 15)}

    @staticmethod
    def _check_required_sensors(data, data_use, bilateral=True):
        """
        Check for the required sensors

        Parameters
        ----------
        data : dict
            Dictionary of input data that will be used for estimating joint angles
        data_use : {'static', 'joint center', 'trial'}
            Use for the data. Either 'static', 'joint center', or 'trial'

        Raises
        ------
        ValueError
            If the required sensors are missing from the dictionary
        """
        if data_use == 'static':
            # required sensors : lumbar, left and right thigh
            req = ['Lumbar', 'Left thigh', 'Right thigh']
            if not all([i in [j for j in data.keys()] for i in req]):
                raise ValueError(f'Static data does not have the required sensors. Ensure it has "Lumbar", '
                                 f'"Left thigh", and "Right thigh" data.')
        elif data_use == 'joint center':
            # required sensors : lumbar, left and right thigh, left and right shank
            req = ['Lumbar', 'Left thigh', 'Right thigh', 'Left shank', 'Right shank']
            if not all([i in [j for j in data.keys()] for i in req]):
                raise ValueError(f'Joint center computation data does not have the required sensors. Ensure it has '
                                 f'"Lumbar", "Left thigh", "Right thigh", "Left shank", and "Right shank" data.')
        elif data_use == 'trial':
            # required sensors : lumbar, left and right thigh
            req = ['Lumbar', 'Left thigh', 'Right thigh']
            if not all([i in [j for j in data.keys()] for i in req]):
                raise ValueError(f'Trial data does not have the required sensors. Ensure it has "Lumbar", '
                                 f'"Left thigh", and "Right thigh" data.')


class OmcAngles:
    def __init__(self, marker_names='default', window=1.0):
        """
        Estimation of hip joint angles from optical motion capture marker trajectories.

        Parameters
        ----------
        marker_names : {default, pykinematics.omc.MarkerNames}, optional
            List of marker names and what they are named in the data provided. Default value is `default`, which
            will use the default values, which can be seen through :meth:`pykinematics.omc.default_marker_names`.
        window : float, optional
            Window size in seconds of the duration to use to find the period of best stillness. Default is 1s.
        """
        self.marker_names = marker_names
        self.window_t = window

        # initialize things used later
        self.pelvis = None
        self.left_thigh = None
        self.right_thigh = None

    def estimate(self, task_data, pelvis_cluster=False):
        """
        Estimate hip joint angles from optical motion capture marker trajectories, following calibration.

        Parameters
        ----------
        task_data : dict
            Dictionary of marker position data for a static standing trial, separated based on segment attached to.
            Requires at least Pelvis, Left thigh, and Right thigh. See *Notes* for the layout.
        pelvis_cluster : bool, optional
            Use the pelvis cluster to anatomical frame to create the anatomical frame for the trial, or create the
            pelvis anatomical frame from the marker trajectories (requires that the markers used to create the frame
            be present for the trial).

        Returns
        -------
        left_hip_angles : numpy.ndarray
            Array of hip angles for the left hip, in the order Flexion - Extension, Ad / Abduction, and
            Internal - External rotation.
        right_hip_angles : numpy.ndarray
            Array of hip angles for the right hip, in the order Flexion - Extension, Ad / Abduction, and
            Internal - External rotation.
        """
        # create the segment frames
        pelvis_af = omc.segmentFrames.pelvis(task_data['Pelvis'], use_cluster=pelvis_cluster, R_s_c=self.pelvis[2],
                                             marker_names=self.marker_names)
        lthigh_af = omc.segmentFrames.thigh(task_data['Left thigh'], 'left', use_cluster=True, R_s_c=self.left_thigh[2],
                                            marker_names=self.marker_names)
        rthigh_af = omc.segmentFrames.thigh(task_data['Right thigh'], 'right', use_cluster=True,
                                            R_s_c=self.right_thigh[2],
                                            marker_names=self.marker_names)

        # compute the joint angles
        l_hip_angles = omc.angles.hip(pelvis_af, lthigh_af, 'left')
        r_hip_angles = omc.angles.hip(pelvis_af, rthigh_af, 'right')

        return l_hip_angles, r_hip_angles

    def calibrate(self, static_data, joint_center_data, fs):
        """
        Calibrate optical motion capture segments based on standard definitions of fixed and floating axes.

        Parameters
        ----------
        static_data : dict
            Dictionary of marker position data for a static standing trial, separated based on segment attached to.
            Requires at least Pelvis, Left thigh, and Right thigh. See *Notes* for the layout.
        joint_center_data : dict
            Dictionary of marker position data for a static standing trial, separated based on segment attached to.
            Requires at least Pelvis, Left thigh, and Right thigh. See *Notes* for the layout.
        fs : float
            Sampling frequency in Hz.

        Attributes
        ----------
        self.pelvis
            3-tuple of pelvis anatomical frame, rotation from world to pelvis cluster frame, and constant rotation from
            segment frame to cluster frame.
        self.left_thigh : tuple
            3-tuple of left thigh anatomical frame, rotation from world to the left thigh cluster frame, and constant
            rotation from segment frame to cluster frame.
        self.right_thigh : tuple
            3-tuple of right thigh anatomical frame, rotation from world to the right thigh cluster frame, and constant
            rotation from segment frame to cluster frame.

        Notes
        -----
        Data dictionary key-level schematic:

        - Pelvis
            - left_asis
            - right_asis
            - ...
        - Left thigh
            - left_lat_femoral_epicondyle
            - ...
        - Right thigh
            - right_lat_femoral_epicondyle
            - ...
        """
        # TODO update omc.calibration.process_static to accept dictionaries. Or change the whole input
        # change to dumb format specified originally
        static_data_tuple = (static_data['Pelvis'], static_data['Left thigh'], static_data['Right thigh'])
        joint_center_data_tuple = (joint_center_data['Pelvis'], joint_center_data['Left thigh'],
                                   joint_center_data['Right thigh'])

        window = int(self.window_t * fs)
        # process the calibration data
        self.pelvis, self.left_thigh, self.right_thigh = omc.calibration.static(static_data_tuple,
                                                                                joint_center_data_tuple,
                                                                                window, self.marker_names)
