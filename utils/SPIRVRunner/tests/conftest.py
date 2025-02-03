# conftest.py
import pytest

# To keep track of the total and passed tests
total_tests = 0
passed_tests = 0


def pytest_runtest_protocol(item, nextitem):
    global total_tests
    total_tests += 1  # Increment total test count
    return None  # Continue with the next tests


def pytest_runtest_makereport(item, call):
    global passed_tests
    if call.when == "call":  # Only consider the final result
        if call.excinfo is None:  # If the test passed
            passed_tests += 1
            status = 'PASS'
        else:  # If the test failed
            status = 'FAIL'
        # Print the result for each test and the percentage
        print(f"Test: {item.nodeid}, Status: {status}")
        percent = (passed_tests / total_tests) * 100
        print(f"Progress: {passed_tests}/{total_tests} tests passed ({percent:.2f}%)")


# Optionally, you could also use a final report to show the overall result.
def pytest_terminal_summary(terminalreporter):
    global passed_tests, total_tests
    percent = (passed_tests / total_tests) * 100
    terminalreporter.write_sep("=", f"Test Results: {passed_tests}/{total_tests} tests passed ({percent:.2f}%)")
