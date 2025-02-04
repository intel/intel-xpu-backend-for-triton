import os
import io
import pytest
import tempfile
import contextlib


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    if os.name == "nt":
        # stub, as pytest_forked doesn't work on windows
        parser.addoption("--forked", action="store_true")


def _run_test_in_subprocess(pyfuncitem_obj, funcargs, child_conn) -> object:
    temp_stdout_buffer = io.StringIO()
    temp_stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(temp_stdout_buffer) as stdout:
        with contextlib.redirect_stderr(temp_stderr_buffer) as stderr:
            res = pyfuncitem_obj(**funcargs)

    child_conn.send({"captured stdout": stdout.getvalue(), "captured stderr": stderr.getvalue()})
    return res


def pytest_pyfunc_call(pyfuncitem: pytest.Function):

    if os.name == "nt" and "forked" in pyfuncitem.keywords:
        from multiprocessing import Process, Pipe
        from pytest import fail

        parent_conn, child_conn = Pipe()
        # Python subprocess tasked with running this test.
        test_subprocess = Process(target=_run_test_in_subprocess,
                                  args=(pyfuncitem.obj, pyfuncitem.funcargs, child_conn))

        test_subprocess.start()
        test_subprocess.join()

        if parent_conn.poll(1):
            print(f"Captured streams from isolated process: '{parent_conn.recv()}'")
        else:
            print("No data sent from isolated process")

        child_conn.close()

        if test_subprocess.exitcode != 0:
            exception_message = (
                f'Test "{pyfuncitem.name}" failed in isolated subprocess with: {test_subprocess.exitcode}')

            # Raise a pytest-compliant exception.
            raise fail(exception_message, pytrace=False)

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
