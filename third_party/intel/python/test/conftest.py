import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache(tmp_path):
    # Use pytest's tmp_path, not tempfile.TemporaryDirectory: on Windows Triton loads
    # the compiled kernel .pyd from knobs.cache.dir via LoadLibrary and the OS holds the
    # handle past the fixture, so same-test teardown deletion raises WinError 5/267.
    # tmp_path defers cleanup to a later pytest session, after the loading process (and
    # its DLL handles) has exited. Verified on BMG XPU.
    from triton import knobs

    with knobs.cache.scope(), knobs.runtime.scope():
        knobs.cache.dir = str(tmp_path)
        yield str(tmp_path)
