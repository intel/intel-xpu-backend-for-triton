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
