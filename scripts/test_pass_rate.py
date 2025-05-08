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
