import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    from triton import knobs
    with knobs.compilation.scope():
        knobs.compilation.always_compile = True
        yield
