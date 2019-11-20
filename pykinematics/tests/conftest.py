import pytest
from numpy import array


def pytest_addoption(parser):
    parser.addoption(
        "--no-integration", action="store_true", default=False, help="Don't run integration testing"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as an integration test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--no-integration'):
        return  # --no-integration wasn't given, run all tests
    skip_integration = pytest.mark.skip(reason="need --no-integration option to not run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def simple_x():
    return array([1, 9, 4, 2, 1, 3, 5, 4, 3, 5, 1, 3, 8, 4, 7, 6, 1, 2])


@pytest.fixture
def ma_window():
    return 5


@pytest.fixture
def simple_x_mov_stats():
    mn = array([0.0, 0.0, 0.0, 3.4, 3.8, 3.0, 3.0, 3.2, 4.0, 3.6, 3.2, 4.0, 4.2, 4.6, 5.6, 5.2, 0.0, 0.0])
    sd = array([0.0, 0.0, 0.0, 3.36154726, 3.1144823, 1.58113883, 1.58113883, 1.4832397, 1.0, 1.67332005, 1.4832397,
                2.64575131, 2.58843582, 2.88097206, 2.07364414, 2.77488739, 0.0, 0.0])
    return mn, sd
