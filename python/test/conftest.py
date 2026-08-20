import os
import pytest
import contextlib


def pytest_configure(config):
    # If pytest-sugar is not active, enable instafail
    if not config.pluginmanager.hasplugin("sugar"):
        config.option.instafail = True

    if os.getenv('TEST_UNSKIP') == 'true':
        # define a function that do nothing
        def unskip(reason=None, allow_module_level=False):
            pass

        # save the original 'pytest.skip' to config._skip_f
        config._skip_f = pytest.skip
        # replace 'pytest.skip' with 'pass' call
        pytest.skip = unskip
    else:
        pass


def pytest_unconfigure(config):
    if os.getenv('TEST_UNSKIP') == 'true':
        # restore 'pytest.skip'
        pytest.skip = config._skip_f
    else:
        pass


@pytest.fixture
def fresh_triton_cache_scope():
    from triton import knobs

    @contextlib.contextmanager
    def fresh_cache():
        with knobs.compilation.scope(), knobs.runtime.scope():
            knobs.compilation.always_compile = True
            yield

    yield fresh_cache


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
