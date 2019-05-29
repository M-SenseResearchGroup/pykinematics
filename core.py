"""
Core functions for computing joint angles from start to finish

Lukas Adamowicz
May 2019
GNU GPL v3.0
"""
from warnings import warn

from pymotion import imu


class ImuAngles:
    def __init__(self, gravity_value):
        """
        Compute angles from MIMU sensors, from initial raw data through joint angles.
        """
        self._grav_val = gravity_value

    def calibrate(self, static_data, joint_center_data):
        # check to ensure that the data provided has the required sensors
        ImuAngles._check_required_sensors(static_data, 'static')
        ImuAngles._check_required_sensors(joint_center_data, 'joint center')

        # get the acceleration scales
        self.acc_scales = dict()
        for sensor in static_data.keys():
            self.acc_scales[sensor] = imu.calibration.get_acc_scale(static_data[sensor]['acceleration'])

        # scale the available data
        for sensor in self.acc_scales.keys():
            static_data[sensor]['Acceleration'] *= self.acc_scales[sensor]

            if sensor in list(joint_center_data.keys()):
                joint_center_data[sensor]['Acceleration'] *= self.acc_scales[sensor]
        # issue a warning if any sensors are in the joint center data but not in the static data scales
        for sens in [sens for sens in joint_center_data.keys() if sens not in list(self.acc_scales.keys())]:
            warn(f'Sensor ({sens}) in joint center data has not been scaled due to no scale factor available from '
                 f'static data provided. Performance may suffer as a result.')



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
            req = ['lumbar', 'left thigh', 'right thigh']
            if not all([i in [j.lower() for j in data.keys()] for i in req]):
                raise ValueError(f'Static data does not have the required sensors. Ensure it has "lumbar", '
                                 f'"left thigh", and "right thigh" data.')
        elif data_use == 'joint center':
            # required sensors : lumbar, left and right thigh, left and right shank
            req = ['lumbar', 'left thigh', 'right thigh', 'left shank', 'right shank']
            if not all([i in [j.lower() for j in data.keys()] for i in req]):
                raise ValueError(f'Joint center computation data does not have the required sensors. Ensure it has '
                                 f'"lumbar", "left thigh", "right thigh", "left shank", and "right shank" data.')
        elif data_use == 'trial':
            # required sensors : lumbar, left and right thigh
            req = ['lumbar', 'left thigh', 'right thigh']
            if not all([i in [j.lower() for j in data.keys()] for i in req]):
                raise ValueError(f'Trial data does not have the required sensors. Ensure it has "lumbar", '
                                 f'"left thigh", and "right thigh" data.')

