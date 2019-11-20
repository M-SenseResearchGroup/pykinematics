import pytest
from numpy import allclose, isclose, array, mean, concatenate

from pykinematics.omc.utility import *


class TestOmcUtility:
    def test_create_cluster_frames(self, lumbar_cluster_sample_data):
        R = create_cluster_frame(lumbar_cluster_sample_data, segment_name='pelvis', marker_names='default')

        assert allclose(R[0], R[1])
        assert allclose(R[0], array([[0.4472136, 0.87287156, -0.19518001],
                                     [0., 0.21821789, 0.97590007],
                                     [0.89442719, -0.43643578, 0.09759001]]))

    @pytest.mark.parametrize(('left_asis', 'right_asis'), ((array([300, 200, 800]), array([0, 200, 650])),
                                                           (array([-150, 325, 920]), array([-430, 25, 330]))))
    def test_compute_pelvis_origin(self, left_asis, right_asis):
        origin = compute_pelvis_origin(left_asis, right_asis)

        gnd = mean(concatenate((left_asis.reshape((-1, 3)), right_asis.reshape((-1, 3))), axis=0), axis=0)
        assert isclose(origin, gnd)

    @pytest.mark.parametrize(('lat_fem_ep', 'med_fem_ep'), ((array([300, 200, 800]), array([0, 200, 650])),
                                                            (array([-150, 325, 920]), array([-430, 25, 330]))))
    def test_compute_thigh_origin(self, lat_fem_ep, med_fem_ep):
        origin = compute_pelvis_origin(lat_fem_ep, med_fem_ep)

        gnd = mean(concatenate((lat_fem_ep.reshape((-1, 3)), med_fem_ep.reshape((-1, 3))), axis=0), axis=0)
        assert isclose(origin, gnd)

    @pytest.mark.parametrize(('lat_mall', 'med_mall'), ((array([300, 200, 800]), array([0, 200, 650])),
                                                        (array([-150, 325, 920]), array([-430, 25, 330]))))
    def test_compute_shank_origin(self, lat_mall, med_mall):
        origin = compute_pelvis_origin(lat_mall, med_mall)

        gnd = mean(concatenate((lat_mall.reshape((-1, 3)), med_mall.reshape((-1, 3))), axis=0), axis=0)
        assert isclose(origin, gnd)
