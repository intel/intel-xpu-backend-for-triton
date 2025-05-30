import os
import sys
import pathlib
import pytest
from typing import Optional, Set
import contextlib


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
    from triton import knobs
    with knobs.compilation.scope():
        knobs.compilation.always_compile = True
        yield


@pytest.fixture
def fresh_triton_cache_scope():
    from triton import knobs

    @contextlib.contextmanager
    def fresh_cache():
        with knobs.compilation.scope():
            knobs.compilation.always_compile = True
            yield

    yield fresh_cache


def _fresh_knobs_impl(monkeypatch, skipped_attr: Optional[Set[str]] = None):
    from triton import knobs

    if skipped_attr is None:
        skipped_attr = set()

    knobs_map = {
        name: knobset
        for name, knobset in knobs.__dict__.items()
        if isinstance(knobset, knobs.base_knobs) and knobset != knobs.base_knobs and name not in skipped_attr
    }

    # We store which variables we need to unset below in finally because
    # monkeypatch doesn't appear to reset variables that were never set
    # before the monkeypatch.delenv call below.
    env_to_unset = []
    prev_propagate_env = knobs.propagate_env

    def fresh_function():
        nonlocal env_to_unset
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset.copy().reset())
            for knob in knobset.knob_descriptors.values():
                if knob.key in os.environ:
                    monkeypatch.delenv(knob.key, raising=False)
                else:
                    env_to_unset.append(knob.key)
        knobs.propagate_env = True
        return knobs

    def reset_function():
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset)
        for k in env_to_unset:
            if k in os.environ:
                del os.environ[k]
        knobs.propagate_env = prev_propagate_env

    return fresh_function, reset_function


@pytest.fixture
def fresh_knobs(monkeypatch):
    fresh_function, reset_function = _fresh_knobs_impl(monkeypatch)
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_except_libraries(monkeypatch):
    """
    A variant of `fresh_knobs` that keeps library path
    information from the environment as these may be
    needed to successfully compile kernels.
    """
    fresh_function, reset_function = _fresh_knobs_impl(monkeypatch, skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())
