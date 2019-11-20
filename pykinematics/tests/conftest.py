import pytest


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