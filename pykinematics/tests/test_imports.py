"""
Testing imports of required libraries
"""


def test_numpy():
    import numpy

    return


def test_scipy():
    import scipy

    return


def test_pykinematics():
    import pykinematics
    from pykinematics import imu
    from pykinematics import omc
    from pykinematics.core import MimuAngles, OmcAngles

    return


def test_pykinematics_imu():
    from pykinematics.imu import angles, calibration, joints, optimize, orientation, utility

    return


def test_pykinematics_omc():
    from pykinematics.omc import angles, calibration, segmentFrames, MarkerNames, default_marker_names

    return
