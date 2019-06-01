"""
Core functions for computing joint angles from start to finish

Lukas Adamowicz
May 2019
GNU GPL v3.0
"""
from numpy import mean, diff
from scipy.signal import butter, filtfilt
from warnings import warn

from pymotion import imu


class ImuAngles:
    def __init__(self, gravity_value=9.81, filter_values=None, angular_velocity_derivative_order=2,
                 joint_center_kwargs=None, orientation_kwargs=None):
        """
        Compute angles from MIMU sensors, from initial raw data through joint angles.
        """
        self.grav_val = gravity_value

        # set the default filter values
        self.set_default_filter_values()
        if filter_values is not None:  # see if any changes to be included
            if isinstance(filter_values, dict):  # check that the format is followed
                for key in filter_values.keys():  # change the default values
                    self.filt_vals[key] = filter_values[key]

        if angular_velocity_derivative_order == 2 or angular_velocity_derivative_order == 4:
            self.wd_ord = angular_velocity_derivative_order
        else:
            raise ValueError('The order of the angular velocity derivative must be either 2 or 4.')

        self.center_kwargs = joint_center_kwargs
        self.orient_kwargs = orientation_kwargs

    def calibrate(self, static_data, joint_center_data):
        # check to ensure that the data provided has the required sensors
        ImuAngles._check_required_sensors(static_data, 'static')
        ImuAngles._check_required_sensors(joint_center_data, 'joint center')

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

        # compute joint centers from the joint center data
        joint_center = imu.joints.Center(g=self.grav_val, **self.center_kwargs)

        # if the center method is "SAC", we need to compute the relative orientation first
        srof = imu.orientation.SROFilter(g=self.grav_val, **self.orient_kwargs)
        if joint_center.method == 'SAC':
            q_lt_p, self.R_lt_lb = ImuAngles._run_orientation(srof, joint_center_data['Lumbar'],
                                                              joint_center_data['Left Thigh'])

    @staticmethod
    def _run_orientation(sro, sensor1, sensor2):
        """
        Run the orientaiton estimation filter. Rotation is provided from sensor 2 -> sensor 1

        Parameters
        ----------
        sro : pymotion.imu.orientation.SROFilter
            Sensor relative orientation estimation object
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
        q = sro.run(sensor1['Acceleration'], sensor2['Acceleration'], sensor1['Angular velocity'],
                    sensor2['Angular velocity'], sensor1['Magnetic field'], sensor2['Magnetic field'])

        R = imu.utility.quat2matrix(q)  # convert to a rotation matrix

        return q, R

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
            ImuAngles._apply_filter(data[sensor]['Acceleration'], data[sensor]['dt'],
                                    self.filt_vals['Acceleration'][0], self.filt_vals['Acceleration'][1])
            # apply the specified filter to the angular velocity
            ImuAngles._apply_filter(data[sensor]['Angular velocity'], data[sensor]['dt'],
                                    self.filt_vals['Angular velocity'][0], self.filt_vals['Angular velocity'][1])
            # apply the specified filter to the magnetic field reading
            ImuAngles._apply_filter(data[sensor]['Magnetic field'], data[sensor]['dt'],
                                    self.filt_vals['Magnetic field'][0], self.filt_vals['Magnetic field'][1])

            if comp_angular_accel:
                data[sensor]['Angular acceleration'] = imu.utility.calc_derivative(data[sensor]['Angular velocity'],
                                                                                   data[sensor]['dt'], order=self.wd_ord)
                ImuAngles._apply_filter(data[sensor]['Angular acceleration'], data[sensor]['dt'],
                                        self.filt_vals['Angular acceleration'][0],
                                        self.filt_vals['Angular acceleration'][0])

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
        b, a = butter(filt_order, filt_cutoff * 2 * dt)
        if x.ndim == 2:
            x = filtfilt(b, a, x, axis=0)
        elif x.ndim == 1:
            x = filtfilt(b, a, x)

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
    def _check_required_sensors(data, data_use):
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

