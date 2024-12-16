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
    # Ignore cleanup errors to avoid PermissionError on Windows: certain .pyd files are locked by
    # Python process and cannot be deleted.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        try:
            os.environ["TRITON_CACHE_DIR"] = tmpdir
            yield tmpdir
        finally:
            os.environ.pop("TRITON_CACHE_DIR", None)
