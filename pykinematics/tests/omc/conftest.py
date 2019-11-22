import pytest
from numpy import array


@pytest.fixture
def lumbar_cluster_sample_data():
    cls1 = array([[500, 400, 800], [-50, 300, 400]], dtype=float)
    cls3 = array([[500, 300, 1800], [-50, 200, 1400]], dtype=float)
    cls2 = array([[0, 300, 800], [-550, 200, 400]], dtype=float)

    return {'sacrum_cluster1': cls1, 'sacrum_cluster2': cls2, 'sacrum_cluster3': cls3}


@pytest.fixture
def lumbar_marker_sample():
    data = {'right_asis': array([[600, 500, 1200], [610, 470, 1200]], dtype=float),
            'left_asis': array([[300, 500, 1200], [310, 500, 1180]], dtype=float),
            'right_psis': array([[500, 300, 1200], [520, 310, 1160]], dtype=float),
            'left_psis': array([[400, 300, 1200], [430, 300, 1150]], dtype=float)}
    return data


@pytest.fixture
def thigh_marker_sample():
    data = {'right_lat_femoral_epicondyle': array([700, 400, 500], dtype=float),
            'right_med_femoral_epicondyle': array([540, 400, 500], dtype=float)}
    return data


@pytest.fixture
def right_hip_jc():
    return array([620, 400, 1100])
