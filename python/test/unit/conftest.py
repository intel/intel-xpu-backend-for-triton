import os
import sys
import pathlib
import pytest
import tempfile


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    if os.name == "nt":
        config.addinivalue_line("markers", "forked: subprocess analogue of pytest.mark.forked on Windows")


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    if os.name == "nt":
        # stub, as pytest_forked doesn't work on windows
        parser.addoption("--forked", action="store_true")


def pytest_pyfunc_call(pyfuncitem: pytest.Function):

    if os.name == "nt" and "forked" in pyfuncitem.keywords:
        # Avoid recursion
        if os.getenv("_PYTEST_SUBPROCESS_RUNNING"):
            return None

        import subprocess

        pos = pyfuncitem.nodeid.find(str(pyfuncitem.path.relative_to(pathlib.Path(os.getcwd()))))
        test_name = pyfuncitem.nodeid[pos:]

        python_executable = sys.executable
        pytest_args = [python_executable, "-m", "pytest", "-s", test_name, "-q"]

        config = pyfuncitem.config
        device = config.getoption("--device")
        if device:
            pytest_args.extend(["--device", device])

        # Avoid recursion
        env = os.environ.copy()
        env["_PYTEST_SUBPROCESS_RUNNING"] = "1"

        print("\n##### start output from pytest in subprocess #####")
        result = subprocess.run(pytest_args, text=True, env=env)
        print("\n##### end output from pytest in subprocess #####")

        if result.returncode != 0:
            # Human-readable exception message to be raised.
            exception_message = (f'Test "{pyfuncitem.name}" failed in isolated subprocess with: {result.returncode}')

            # Raise a pytest-compliant exception.
            raise pytest.fail(exception_message, pytrace=False)

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


@pytest.fixture
def fresh_config(request, monkeypatch):
    from triton import config
    config_map = {
        name: conf
        for name, conf in config.__dict__.items()
        if isinstance(conf, config.base_config) and conf != config.base_config
    }
    try:
        for name, conf in config_map.items():
            setattr(config, name, conf.copy().reset())
            for knob in conf.knob_descriptors.values():
                monkeypatch.delenv(knob.key, raising=False)
        yield config
    finally:
        for name, conf in config_map.items():
            setattr(config, name, conf)
