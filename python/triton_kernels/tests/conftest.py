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
        gpu_id = int(worker_id[2:])  # gw0 → 0, gw1 → 1, etc.
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id % torch.cuda.device_count())

        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/worker_{worker_id}.log"

        # open file for both stdout/stderr
        log_fd = open(log_path, "w", buffering=1)

        # redirect Python-level stdout/stderr
        sys.stdout = log_fd
        sys.stderr = log_fd

        # redirect C-level stdout/stderr (e.g. std::cout, printf)
        os.dup2(log_fd.fileno(), 1)  # stdout
        os.dup2(log_fd.fileno(), 2)  # stderr

        # flush buffers
        try:
            libc = ctypes.CDLL(None)
            libc.fflush(None)
        except Exception:
            pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
