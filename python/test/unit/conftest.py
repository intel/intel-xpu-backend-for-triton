import os
import pytest
import tempfile


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.environ["TRITON_CACHE_DIR"] = tmpdir
                yield tmpdir
            finally:
                os.environ.pop("TRITON_CACHE_DIR", None)
    except OSError:
        # Ignore errors, such as PermissionError, on Windows
        pass


def pytest_configure(config):
    worker_id = os.getenv("PYTEST_XDIST_WORKER")
    # On Windows, use a dedicated Triton cache per pytest worker to avoid PermissionError.
    if os.name == "nt" and worker_id:
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-")
