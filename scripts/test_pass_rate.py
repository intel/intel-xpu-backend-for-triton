import pass_rate

WARNINGS = """\
[
    {"location": "message"}
]
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
