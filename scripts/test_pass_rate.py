from xml.etree.ElementTree import parse

import pass_rate

WARNINGS = """\
[
    {"location": "message"}
]
"""

PYTEST_SELECT_WARNING_1 = """\
plugin.py:100: UserWarning: pytest-skip: Not all deselected tests exist (or have been selected otherwise).
Missing deselected test names:
  - test11
  - test12
"""

PYTEST_SELECT_WARNING_2 = """\
plugin.py:100: UserWarning: pytest-skip: Not all deselected tests exist (or have been selected otherwise).
Missing deselected test names:
  - test21
  - test22
"""

# Real JUnit report with the 3 tests. One test passed, but there is an error at test teardown.
# Note that pytest incorrectly reports 4 tests instead of 3.
# https://github.com/intel/intel-xpu-backend-for-triton/issues/4341.
JUNIT_XML_REPORT1 = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="1" failures="0" skipped="2" tests="4" time="31.615" timestamp="2025-05-28T15:26:42.685704+00:00" hostname="example">
    <testcase classname="python.test.unit.test_perf_warning" name="test_remark_swp_op_before_operands" time="17.602"><error message="xxx"></error></testcase>
    <testcase classname="python.test.unit.test_perf_warning" name="test_mma_remark" time="0.019"><skipped type="pytest.xfail" message="Not designed for XPU" /></testcase>
    <testcase classname="python.test.unit.test_perf_warning" name="test_remark_vectorization" time="0.030"><skipped type="pytest.xfail" message="Not designed for XPU" /></testcase>
  </testsuite>
</testsuites>
"""

# JUnit report with the same test reported twice: one for test failure, another for error at teardown
# https://github.com/intel/intel-xpu-backend-for-triton/issues/4331.
JUNIT_XML_REPORT2 = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite>
    <testcase classname="python.test.unit.runtime.test_compilation_listener" name="test_compile_stats">
      <failure/>
    </testcase>
    <testcase classname="python.test.unit.runtime.test_compilation_listener" name="test_compile_stats">
      <error/>
    </testcase>
  </testsuite>
</testsuites>
"""



def test_get_warnings_empty_file(tmp_path):
    warnings_path = tmp_path / 'suite-warnings.json'
    warnings_path.write_text('[]')
    assert pass_rate.get_warnings(tmp_path, 'suite') == []


def test_get_warnings(tmp_path):
    warnings_path = tmp_path / 'suite-warnings.json'
    warnings_path.write_text(WARNINGS)
    warnings = pass_rate.get_warnings(tmp_path, 'suite')
    assert len(warnings) == 1
    assert warnings[0].location == 'location'
    assert warnings[0].message == 'message'


def test_generate_junit_report(tmp_path):
    file1 = tmp_path / 'tutorial-1.json'
    file2 = tmp_path / 'tutorial-2.json'
    file3 = tmp_path / 'tutorial-3.json'

    file1.write_text('{"name": "1", "result": "PASS", "time": 1.0}')
    file2.write_text('{"name": "2", "result": "SKIP"}')
    file3.write_text('{"name": "3", "result": "FAIL", "message": "Error"}')

    pass_rate.generate_junit_report(tmp_path)

    report_path = tmp_path / 'tutorials.xml'
    assert report_path.exists()

    xml = parse(report_path)
    testsuites = xml.getroot()
    assert testsuites.tag == 'testsuites'

    testsuite = testsuites[0]
    assert testsuite.tag == 'testsuite'
    assert testsuite.get('name') == 'tutorials'
    assert testsuite.get('tests') == '3'
    assert testsuite.get('skipped') == '1'
    assert testsuite.get('failures') == '1'

    stats = pass_rate.parse_junit_reports(tmp_path)
    assert len(stats) == 1
    assert stats[0].name == 'tutorials'
    assert stats[0].passed == 1
    assert stats[0].failed == 1
    assert stats[0].skipped == 1
    assert stats[0].total == 3


def test_parse_report_ignore_tests_attribute(tmp_path):
    report_path = tmp_path / 'suite.xml'
    report_path.write_text(JUNIT_XML_REPORT1)
    stats = pass_rate.parse_report(report_path)
    assert stats.passed == 0
    assert stats.xfailed == 2
    assert stats.failed == 1
    assert stats.skipped == 0
    assert stats.total == 3


def test_parse_report_duplicate_test(tmp_path):
    report_path = tmp_path / 'suite.xml'
    report_path.write_text(JUNIT_XML_REPORT2)
    stats = pass_rate.parse_report(report_path)
    assert stats.passed == 0
    assert stats.xfailed == 0
    assert stats.failed == 1
    assert stats.skipped == 0
    assert stats.total == 1
