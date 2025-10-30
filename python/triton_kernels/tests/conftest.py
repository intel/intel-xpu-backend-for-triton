import pytest
import tempfile
import os
import sys
import ctypes


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None and worker_id.startswith("gw"):
        import torch
        gpu_id = int(worker_id[2:])  # map gw0 → 0, gw1 → 1, ...
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id % torch.cuda.device_count())


        os.makedirs("logs", exist_ok=True)
        stdout_path = f"logs/stdout_{worker_id}.log"
        stderr_path = f"logs/stderr_{worker_id}.log"

        stdout_fd = open(stdout_path, "w", buffering=1)
        stderr_fd = open(stderr_path, "w", buffering=1)

        sys.stdout = stdout_fd
        sys.stderr = stderr_fd
        os.dup2(stdout_fd.fileno(), 1)
        os.dup2(stderr_fd.fileno(), 2)

        try:
            libc = ctypes.CDLL(None)
            libc.fflush(None)  # flush all C stdio buffers
        except Exception:
            pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

