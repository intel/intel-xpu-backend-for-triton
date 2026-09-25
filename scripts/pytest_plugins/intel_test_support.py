"""Intel-only pytest support, kept out of the shared conftest files.

Loaded by `scripts/pytest-utils.sh` (`-p intel_test_support`), so the shared
`python/test/conftest.py` stays identical to upstream.

TEST_UNSKIP=true turns `pytest.skip()` into a no-op, which lets a CI run report what
the skipped tests actually do instead of hiding them. It is a debugging mode: Intel's
own harness pairs it with TRITON_TEST_IGNORE_ERRORS=true.
"""
import os

import pytest

# Where the original pytest.skip is parked while it is patched out. A stash key keeps that
# state on the session's own config object, which is what pytest offers for plugin state.
_SAVED_SKIP = pytest.StashKey[object]()


def _unskip(reason=None, allow_module_level=False):  # pylint: disable=unused-argument
    """Replacement for pytest.skip() that does nothing.

    The signature has to mirror pytest.skip, because callers pass both arguments by keyword;
    they are accepted and ignored rather than renamed.
    """


# Tests catch the skip exception as `pytest.skip.Exception` (see
# python/triton_kernels/tests/test_matmul.py), so the replacement has to expose it too -
# otherwise, once an exception reaches such an except clause, looking the attribute up on a
# bare function raises AttributeError and hides the failure that was being reported.
_unskip.Exception = pytest.skip.Exception  # type: ignore[attr-defined]


def pytest_configure(config):
    if os.getenv("TEST_UNSKIP") == "true":
        config.stash[_SAVED_SKIP] = pytest.skip
        pytest.skip = _unskip


def pytest_unconfigure(config):
    saved = config.stash.get(_SAVED_SKIP, None)
    if saved is not None:
        pytest.skip = saved
