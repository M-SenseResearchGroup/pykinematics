"""
Core functions for computing joint angles from start to finish

Lukas Adamowicz
May 2019
GNU GPL v3.0
"""
from pymotion import imu


class ImuAngles:
    def __init__(self, gravity_value):
        """
        Compute angles from MIMU sensors, from initial raw data through joint angles.
        """
        self._grav_val = gravity_value

    def run(self, static_data, joint_center_data, trial_data):
        imu.calibration.get_acc_scale()

