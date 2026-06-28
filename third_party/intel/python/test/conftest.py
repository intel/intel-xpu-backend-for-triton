import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    import tempfile
    from triton import knobs
    with tempfile.TemporaryDirectory() as tmpdir:
        with knobs.cache.scope(), knobs.compilation.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            knobs.compilation.always_compile = True
            yield tmpdir
