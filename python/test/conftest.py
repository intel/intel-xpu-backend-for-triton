import pytest
import contextlib


def pytest_configure(config):
    # If pytest-sugar is not active, enable instafail
    if not config.pluginmanager.hasplugin("sugar"):
        config.option.instafail = True


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
