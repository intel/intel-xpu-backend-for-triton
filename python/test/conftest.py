# content of conftest.py
import os
import pytest
import setproctitle

def pytest_runtest_protocol(item, nextitem):
    """
    Hook to set the process title to the name of the currently running test.
    """
    # Set the process title to the current test name
    test_name = item.nodeid
    setproctitle.setproctitle(f"pytest: {test_name}")

    # Run the test
    outcome = yield

    # Reset the process title after the test is done
    setproctitle.setproctitle("pytest: idle")

    return outcome

def pytest_configure(config):
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
