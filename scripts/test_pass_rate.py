import json

import pass_rate

WARNINGS = """\
[
    {"location": "message"}
]
"""

PYTEST_SELECT_WARNING_1 = """\
plugin.py:100: PytestSelectWarning: pytest-select: Not all deselected tests exist (or have been selected otherwise).
Missing deselected test names:
  - test11
  - test12
"""

PYTEST_SELECT_WARNING_2 = """\
plugin.py:100: PytestSelectWarning: pytest-select: Not all deselected tests exist (or have been selected otherwise).
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


def test_get_missing_tests():
    warnings = [
        pass_rate.TestWarning(location='', message='Warning1'),
        pass_rate.TestWarning(location='', message=PYTEST_SELECT_WARNING_2),
        pass_rate.TestWarning(location='', message='Warning2'),
        pass_rate.TestWarning(location='', message=PYTEST_SELECT_WARNING_1),
    ]
    tests = pass_rate.get_missing_tests(warnings)
    assert tests == ['test11', 'test12', 'test21', 'test22']


def test_get_all_missing_tests(tmp_path):
    suite1_path = tmp_path / 'suite1.xml'
    warnings1_path = tmp_path / 'suite1-warnings.json'
    warnings1 = [
        {'': 'warning1'},
        {'': PYTEST_SELECT_WARNING_1},
        {'': PYTEST_SELECT_WARNING_1},
    ]
    suite1_path.write_text('')
    warnings1_path.write_text(json.dumps(warnings1))

    suite2_path = tmp_path / 'suite2.xml'
    warnings2_path = tmp_path / 'suite2-warnings.json'
    warnings2 = [
        {'': 'warning2'},
        {'': PYTEST_SELECT_WARNING_2},
        {'': PYTEST_SELECT_WARNING_2},
    ]
    suite2_path.write_text('')
    warnings2_path.write_text(json.dumps(warnings2))

    tests = pass_rate.get_all_missing_tests(tmp_path)
    assert tests == {
        'suite1': ['test11', 'test12'],
        'suite2': ['test21', 'test22'],
    }
