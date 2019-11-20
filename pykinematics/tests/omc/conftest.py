import pytest
from numpy import array


@pytest.fixture
def lumbar_cluster_sample_data():
    cls1 = array([[500, 400, 800], [-50, 300, 400]], dtype=float)
    cls3 = array([[500, 300, 1800], [-50, 200, 1400]], dtype=float)
    cls2 = array([[0, 300, 800], [-550, 200, 400]], dtype=float)

    return {'sacrum_cluster1': cls1, 'sacrum_cluster2': cls2, 'sacrum_cluster3': cls3}
