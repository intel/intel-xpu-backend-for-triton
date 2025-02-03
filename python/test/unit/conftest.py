import os
import sys
import pytest
import tempfile


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    if os.name == "nt":
        # stub, as pytest_forked doesn't work on windows
        parser.addoption("--forked", action="store_true")


class StdStreamWrapper(object):

    def __init__(self, conn) -> None:
        self.conn = conn

    def write(self, data) -> None:
        self.conn.send(data)
        self.conn.send(data)

    def writelines(self, datas) -> None:
        self.conn.send(datas)

    def __getattr__(self, attr: str) -> object:
        return getattr(self.conn, attr)


def _run_test_in_subprocess(pyfuncitem_obj, child_conn) -> object:
    sys.stderr = StdStreamWrapper(child_conn)
    sys.stdout = StdStreamWrapper(child_conn)

    # Run this test and return the result of doing so.
    res = pyfuncitem_obj()
    child_conn.close()
    return res


def pytest_pyfunc_call(pyfuncitem: pytest.Function):

    if os.name == "nt" and "forked" in pyfuncitem.keywords:
        # Defer hook-specific imports.
        from multiprocessing import Process, Pipe

        parent_conn, child_conn = Pipe()
        # Python subprocess tasked with running this test.
        test_subprocess = Process(target=_run_test_in_subprocess, args=(pyfuncitem.obj, child_conn))

        # Begin running this test in this subprocess.
        test_subprocess.start()

        # Block this parent Python process until this test completes.
        test_subprocess.join()
        child_conn.close()

        print(f"stdout+stderr from isolated process: '{parent_conn.recv()}'")

        # If this subprocess reports non-zero exit status, this test failed. In
        # this case...
        if test_subprocess.exitcode != 0:
            # Human-readable exception message to be raised.
            exception_message = (
                f'Test "{pyfuncitem.name}" failed in isolated subprocess with: {test_subprocess.exitcode}')

            # Raise a pytest-compliant exception.
            raise pytest.fail(exception_message, pytrace=False)
        # Else, this subprocess reports zero exit status. In this case, this
        # test succeeded.

        # Notify pytest that this hook successfully ran this test.
        return True

    # Notify pytest that this hook avoided attempting to run this test, in which
    # case pytest will continue to look for a suitable runner for this test.
    return None


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["TRITON_CACHE_DIR"] = tmpdir
            yield tmpdir
        finally:
            os.environ.pop("TRITON_CACHE_DIR", None)


def pytest_configure(config):
    worker_id = os.getenv("PYTEST_XDIST_WORKER")
    # On Windows, use a dedicated Triton cache per pytest worker to avoid PermissionError.
    if os.name == "nt" and worker_id:
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-")
