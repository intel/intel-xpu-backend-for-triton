# content of conftest.py
import os
import pytest


def pytest_configure(config):
    if os.getenv('TEST_UNSKIP') == 'false':
        pass
    else:
        # define a function that do nothing
        def unskip(reason=None, allow_module_level=False):
            pass

        # save the original 'pytest.skip' to config._skip_f
        config._skip_f = pytest.skip
        # replace 'pytest.skip' with 'pass' call
        pytest.skip = unskip


def pytest_unconfigure(config):
    if os.getenv('TEST_UNSKIP') == 'false':
        pass
    else:
        # restore 'pytest.skip'
        pytest.skip = config._skip_f
