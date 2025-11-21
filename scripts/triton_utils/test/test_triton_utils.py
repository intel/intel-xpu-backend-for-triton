import dataclasses
from typing import Any, cast

import json
import re

import pytest

import triton_utils

# TODO:  # pylint: disable=fixme
# test list failure reasons skips should not be taken into the account
# test sort by


def extract_json_substrings(s: str) -> list[dict[str, Any] | list[Any]]:
    dec = json.JSONDecoder()
    results: list[dict[str, Any] | list[Any]] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in '{[':
            try:
                val, end = dec.raw_decode(s, i)
            except json.JSONDecodeError:
                i += 1
                continue
            results.append(val)
            i = end
        else:
            i += 1
    return results


def extract_pass_rate_dicts(s: str) -> list[dict[str, Any]]:
    out_jsons = extract_json_substrings(s)
    assert len(out_jsons) > 0
    return [cast(dict[str, Any], out_jsons) for out_jsons in out_jsons]


def extract_pass_rate_dict(s: str) -> dict[str, Any]:
    pass_rate_dicts = extract_pass_rate_dicts(s)
    assert len(pass_rate_dicts) == 1
    return pass_rate_dicts[0]


TESTS_WITH_MULTIPLE_TESTSUITES = {
    'language.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1" time="0.114" timestamp="2024-04-05T11:03:23.033702" hostname="hostname">
        <testcase classname="test.unit.language.test_core" name="test_reduce_layouts[sum-int32-reduce2d-1-src_layout8-32-128]" time="0.114" />
    </testsuite>
</testsuites>
''', 'interpreter.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1" time="0.114" timestamp="2024-04-05T11:03:23.033702" hostname="hostname">
        <testcase classname="test.unit.language.test_core" name="test_reduce_layouts[sum-int32-reduce2d-1-src_layout8-32-128]" time="0.114" />
    </testsuite>
</testsuites>
'''
}


# yapf: disable
@pytest.mark.parametrize(
    ('tests_with_multiple_testsuites', 'ignore_testsuite_filter', 'passed', 'success'),
    [
        (False, [], 0, False),
        (True, [], 2, True),
        (False, ['interpreter'], 1, True),
    ]
)
def test_tests_with_multiple_test_suites(  # pylint: disable=R0913, R0917
    tmp_path,
    tests_with_multiple_testsuites,
    ignore_testsuite_filter,
    passed,
    success,
):
    rep1 = tmp_path / 'language.xml'
    rep1.write_text(TESTS_WITH_MULTIPLE_TESTSUITES['language.xml'], encoding='utf-8')
    rep2 = tmp_path / 'interpreter.xml'
    rep2.write_text(TESTS_WITH_MULTIPLE_TESTSUITES['interpreter.xml'], encoding='utf-8')

    config = triton_utils.Config(
        action='pass_rate',
        reports=str(tmp_path),
        tests_with_multiple_testsuites=tests_with_multiple_testsuites,
        ignore_testsuite_filter=ignore_testsuite_filter,
    )

    try:
        out, _ = triton_utils.PassRateActionRunner(config)()
        assert success
        pass_rate_dict = extract_pass_rate_dict(out)
        assert pass_rate_dict['passed'] == passed
    except ValueError:
        assert not success


TESTS_WITH_DIFFERENT_STATUSES = {
    'language.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="1" failures="1" skipped="2" tests="5" time="86.361" timestamp="2024-04-05T11:03:23.033702" hostname="hostname">
        <testcase classname="python.test.unit.language.test_core" name="test_reduce_layouts[sum-int32-reduce2d-1-src_layout8-32-128]" time="0.114" />
        <testcase classname="python.test.unit.language.test_tensor_descriptor" name="test_tensor_descriptor_reduce[1-1024-device-2-uint16-or]" time="0.001">
            <skipped type="pytest.xfail" message="Multi-CTA not supported" />
        </testcase>
        <testcase classname="python.test.unit.language.test_tensor_descriptor" name="test_host_tensor_descriptor_matmul[128-128-16-4-2]" time="0.000">
            <skipped type="pytest.skip" message="Skipped by pytest-skip">
                /runner/.../test_tensor_descriptor.py:1708: Skipped by pytest-skip
            </skipped>
        </testcase>
        <testcase classname="python.test.unit.language.test_core" name="test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]" time="0.001">
            <error message="failed on setup with &quot;worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'&quot;">
                worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'
            </error>
        </testcase>
        <testcase classname="python.test.unit.language.test_matmul" name="test_blocked_scale_mxfp[False-4-128-128-256-1024-512-512]" time="86.245">
            <failure message="AssertionError: Tensor-likes are not close! ...">
                M = 1024, N = 512, K = 512, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 256
                language/test_matmul.py:859: AssertionError
            </failure>
        </testcase>
    </testsuite>
</testsuites>
''', 'scaled_dot.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests">
    <testsuite name="pytest" errors="0" failures="0" skipped="2" tests="3" time="12.498" timestamp="2025-10-22T16:53:55.914818+00:00" hostname="intel-tools.intel-xpu-backend-for-triton-jgs-testing-w2mstzhg7m">
        <testcase classname="python.test.unit.language.test_matmul" name="test_mxfp8_mxfp4_matmul[0-False-False-float4-float4-False-False-False-3-128-64-128-1024-512-512]" time="0.000">
            <skipped type="pytest.skip" message="Skipped by pytest-skip">
                /runner/_work/.../test_matmul.py:1204: Skipped by pytest-skip
            </skipped>
        </testcase>
        <testcase classname="python.test.unit.language.test_core" name="test_scaled_dot[32-32-64-False-True-False-e2m1-fp16-4-16-1]" time="12.947" />
        <testcase classname="python.test.unit.language.test_matmul" name="test_preshuffle_scale_mxfp_cdna4[True-32-mxfp4-mxfp4-False-32-32-64-1024-1024-1024]" time="0.001">
            <skipped type="pytest.xfail" message="Minimal tile size is .." />
        </testcase>
    </testsuite>
</testsuites>
'''
}


def test_error_on_failures_flag(tmp_path):
    rep1 = tmp_path / 'language.xml'
    rep1.write_text(TESTS_WITH_DIFFERENT_STATUSES['language.xml'], encoding='utf-8')

    config = dataclasses.replace(
        triton_utils.Config(),
        action='pass_rate',
        reports=str(tmp_path),
        error_on_failures=True,
    )

    try:
        triton_utils.run(config)
        assert False
    except SystemExit as e:
        assert e.code == 1


def test_export_to_csv(tmp_path):
    rep1 = tmp_path / 'language.xml'
    rep1.write_text(TESTS_WITH_DIFFERENT_STATUSES['language.xml'], encoding='utf-8')

    config = dataclasses.replace(
        triton_utils.Config(),
        action='export_to',
        reports=str(tmp_path),
    )

    try:
        triton_utils.run(config)
    except SystemExit as e:
        assert e.code == 0


def test_multiple_test_suites(tmp_path):
    rep_language = tmp_path / 'language.xml'
    rep_language.write_text(TESTS_WITH_DIFFERENT_STATUSES['language.xml'], encoding='utf-8')

    rep_scaled_dot = tmp_path / 'scaled_dot.xml'
    rep_scaled_dot.write_text(TESTS_WITH_DIFFERENT_STATUSES['scaled_dot.xml'], encoding='utf-8')

    config = triton_utils.Config(
        action='tests_stats',
        reports=str(tmp_path),
        tests_with_multiple_testsuites=True,
        _report_grouping_level=triton_utils.TestGroupingLevel.TESTSUITE.value,
    )

    out = triton_utils.TestsStatsActionRunner(config=config)()

    pass_rate_dicts = extract_pass_rate_dicts(out)

    assert len(pass_rate_dicts) == 2
    assert pass_rate_dicts[0]['language']['total'] == 5
    assert pass_rate_dicts[1]['scaled_dot']['total'] == 3


@pytest.mark.parametrize(
    ('status_filter', 'passed', 'skipped', 'xfailed', 'failed'),
    [
        ([], 1, 1, 1, 2),
        (['passed'], 1, 0, 0, 0),
        (['skipped'], 0, 1, 0, 0),
        (['xfailed'], 0, 0, 1, 0),
        (['failed'], 0, 0, 0, 2),
        (['passed', 'failed'], 1, 0, 0, 2),
    ]
)
def test_status_filter(  # pylint: disable=R0913, R0917
        tmp_path,
        status_filter,
        passed,
        skipped,
        xfailed,
        failed,
):
    test_rep_path = tmp_path / 'language.xml'
    test_rep_path.write_text(TESTS_WITH_DIFFERENT_STATUSES['language.xml'], encoding='utf-8')

    if status_filter:
        config = triton_utils.Config(
            action='pass_rate',
            reports=str(tmp_path),
            status_filter=status_filter,
        )
    else:
        config = triton_utils.Config(
            action='pass_rate',
            reports=str(tmp_path),
        )

    out, _ = triton_utils.PassRateActionRunner(config)()
    pass_rate_dict = extract_pass_rate_dict(out)
    assert pass_rate_dict['passed'] == passed
    assert pass_rate_dict['skipped'] == skipped
    assert pass_rate_dict['xfailed'] == xfailed
    assert pass_rate_dict['failed'] == failed


@pytest.mark.parametrize(
    ('layout', 'include_subdir_patterns', 'passed', 'skipped', 'xfailed', 'failed', 'empty_subdirs'),
    [
        (
            {'language.xml': 'subdir', 'scaled_dot.xml': 'subdir2'},
            [re.compile(r'^.*$')],
            2, 2, 2, 2, False
        ),
        (
            {'language.xml': 'subdir', 'scaled_dot.xml': 'subdir2'},
            [re.compile(r'^.*$')],
            2, 2, 2, 2, True
        ),
        (
            {'language.xml': 'subdir', 'scaled_dot.xml': 'subdir'},
            [re.compile(r'^.*$')],
            2, 2, 2, 2, False
        ),
        (
            {'language.xml': '', 'scaled_dot.xml': 'subdir'},
            [re.compile(r'^.*$')],
            2, 2, 2, 2, False
        ),
        (
            {'language.xml': '', 'scaled_dot.xml': ''},
            [],
            2, 2, 2, 2, False
        ),
    ]
)
def test_report_directory_layout(  # pylint: disable=R0913, R0914, R0917
        capsys,
        tmp_path,
        layout,
        include_subdir_patterns,
        passed,
        skipped,
        xfailed,
        failed,
        empty_subdirs
):
    for report_name, subdir in layout.items():
        report_path = tmp_path
        if subdir:
            report_path = report_path / subdir
            report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / report_name
        report_file.write_text(TESTS_WITH_DIFFERENT_STATUSES[report_name], encoding='utf-8')

    if empty_subdirs:
        empty_subdir_path = tmp_path / 'empty_subdir'
        empty_subdir_path.mkdir(parents=True, exist_ok=True)

    if include_subdir_patterns:
        config = triton_utils.Config(
            action='pass_rate',
            reports=str(tmp_path),
            include_subdir_patterns=include_subdir_patterns,
        )
    else:
        config = triton_utils.Config(
            action='pass_rate',
            reports=str(tmp_path),
        )

    out, _ = triton_utils.PassRateActionRunner(config)()
    std_out, _ = capsys.readouterr()
    pass_rate_dict = extract_pass_rate_dict(out)
    assert pass_rate_dict['passed'] == passed
    assert pass_rate_dict['skipped'] == skipped
    assert pass_rate_dict['xfailed'] == xfailed
    assert pass_rate_dict['failed'] == failed

    if empty_subdirs:
        assert 'WARNING: No junit xml files' in std_out


@pytest.mark.parametrize(
    ('layout', 'exclude_subdir_patterns', 'include_subdir_patterns', 'passed', 'skipped', 'xfailed', 'failed'),
    [
        (
            {'language.xml': 'subdir', 'scaled_dot.xml': 'subdir2'},
            [re.compile(r'subdir2')],
            [re.compile(r'^.*$')],
            1, 1, 1, 2,
        ),
    ]
)
def test_report_directory_patterns(  # pylint: disable=R0913, R0917
        tmp_path,
        layout,
        exclude_subdir_patterns,
        include_subdir_patterns,
        passed,
        skipped,
        xfailed,
        failed,
):
    for report_name, subdir in layout.items():
        report_path = tmp_path
        if subdir:
            report_path = report_path / subdir
            report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / report_name
        report_file.write_text(TESTS_WITH_DIFFERENT_STATUSES[report_name], encoding='utf-8')

    config = triton_utils.Config(
        action='pass_rate',
        reports=str(tmp_path),
        exclude_subdir_patterns=exclude_subdir_patterns,
        include_subdir_patterns=include_subdir_patterns,
    )

    out, _ = triton_utils.PassRateActionRunner(config)()
    pass_rate_dict = extract_pass_rate_dict(out)
    assert pass_rate_dict['passed'] == passed
    assert pass_rate_dict['skipped'] == skipped
    assert pass_rate_dict['xfailed'] == xfailed
    assert pass_rate_dict['failed'] == failed


TESTS_WITH_MULTIPLE_RESULTS = {
    'test-report1/language.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="1" failures="1" skipped="2" tests="5" time="86.361" timestamp="2024-04-05T11:03:23.033702" hostname="hostname">
        <testcase classname="python.test.unit.language.test_core" name="test_reduce_layouts[sum-int32-reduce2d-1-src_layout8-32-128]" time="0.114" />
        <testcase classname="python.test.unit.language.test_tensor_descriptor" name="test_tensor_descriptor_reduce[1-1024-device-2-uint16-or]" time="0.001">
            <skipped type="pytest.xfail" message="Multi-CTA not supported" />
        </testcase>
        <testcase classname="python.test.unit.language.test_tensor_descriptor" name="test_host_tensor_descriptor_matmul[128-128-16-4-2]" time="0.000">
            <skipped type="pytest.skip" message="Skipped by pytest-skip">
                /runner/.../test_tensor_descriptor.py:1708: Skipped by pytest-skip
            </skipped>
        </testcase>
        <testcase classname="python.test.unit.language.test_core" name="test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]" time="0.001">
            <error message="failed on setup with &quot;worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'&quot;">
                worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'
            </error>
        </testcase>
        <testcase classname="python.test.unit.language.test_matmul" name="test_blocked_scale_mxfp[False-4-128-128-256-1024-512-512]" time="86.245">
            <failure message="AssertionError: Tensor-likes are not close! ...">
                M = 1024, N = 512, K = 512, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 256
                language/test_matmul.py:859: AssertionError
            </failure>
        </testcase>
    </testsuite>
</testsuites>
''', 'test-report2/language.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="1" failures="1" skipped="3" tests="5" time="86.361" timestamp="2024-04-05T11:03:23.033702" hostname="hostname">
        <testcase classname="python.test.unit.language.test_core" name="test_reduce_layouts[sum-int32-reduce2d-1-src_layout8-32-128]" time="0.114">
            <skipped type="pytest.skip" message="Skipped by pytest-skip">
                /runner/.../test_tensor_descriptor.py:1708: Skipped by pytest-skip
            </skipped>
        </testcase>
        <testcase classname="python.test.unit.language.test_tensor_descriptor" name="test_tensor_descriptor_reduce[1-1024-device-2-uint16-or]" time="0.001">
            <skipped type="pytest.xfail" message="Multi-CTA not supported" />
        </testcase>
        <testcase classname="python.test.unit.language.test_tensor_descriptor" name="test_host_tensor_descriptor_matmul[128-128-16-4-2]" time="0.000">
            <skipped type="pytest.skip" message="Skipped by pytest-skip">
                /runner/.../test_tensor_descriptor.py:1708: Skipped by pytest-skip
            </skipped>
        </testcase>
        <testcase classname="python.test.unit.language.test_core" name="test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]" time="0.001">
            <error message="failed on setup with &quot;worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'&quot;">
                worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'
            </error>
        </testcase>
        <testcase classname="python.test.unit.language.test_matmul" name="test_blocked_scale_mxfp[False-4-128-128-256-1024-512-512]" time="86.245">
            <failure message="AssertionError: Tensor-likes are not close! ...">
                M = 1024, N = 512, K = 512, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 256
                language/test_matmul.py:859: AssertionError
            </failure>
        </testcase>
    </testsuite>
</testsuites>
'''
}


@pytest.mark.parametrize(
    ('merge_test_results', 'exclude_subdir_patterns', 'include_subdir_patterns', 'passed', 'skipped', 'xfailed', 'failed', 'warning'),
    [
        (
            False,
            [re.compile(r'subdir2')],
            [re.compile(r'^.*$')],
            1, 3, 2, 4, True,
        ),
        (
            True,
            [re.compile(r'subdir2')],
            [re.compile(r'^.*$')],
            1, 1, 1, 2, False,
        ),
    ]
)
def test_tests_with_multiple_results(  # pylint: disable=R0913, R0914, R0917
    capsys,
    tmp_path,
    merge_test_results,
    exclude_subdir_patterns,
    include_subdir_patterns,
    passed: int,
    skipped: int,
    xfailed: int,
    failed: int,
    warning: bool,
):
    for report_name, doc_str in TESTS_WITH_MULTIPLE_RESULTS.items():
        report_name_parts = report_name.split('/')
        subdir_path = tmp_path / report_name_parts[0]
        subdir_path.mkdir(parents=True, exist_ok=True)
        report_file = subdir_path / report_name_parts[1]
        report_file.write_text(doc_str, encoding='utf-8')

    config = triton_utils.Config(
        action='pass_rate',
        merge_test_results=merge_test_results,
        exclude_subdir_patterns=exclude_subdir_patterns,
        include_subdir_patterns=include_subdir_patterns,
        reports=str(tmp_path),
    )

    out, _ = triton_utils.PassRateActionRunner(config)()
    std_out, _ = capsys.readouterr()
    pass_rate_dict = extract_pass_rate_dict(out)
    assert pass_rate_dict['passed'] == passed
    assert pass_rate_dict['skipped'] == skipped
    assert pass_rate_dict['xfailed'] == xfailed
    assert pass_rate_dict['failed'] == failed

    if warning:
        assert '[WARNING] Multiple test results for the same test case have been found.' in std_out


@pytest.mark.parametrize(
    ('args', 'config'),
    [
        (
            'pass_rate --reports /path/to/reports',
            triton_utils.Config(
                action='pass_rate',
                reports='/path/to/reports',
            )
        ),
        (
            'pass_rate --reports /path/to/reports --status skipped',
            triton_utils.Config(
                action='pass_rate',
                reports='/path/to/reports',
                status_filter=['skipped'],
            )
        ),
        (
            'tests_stats --r /path/to/reports --status skipped',
            triton_utils.Config(
                action='tests_stats',
                reports='/path/to/reports',
                status_filter=['skipped'],
            )
        ),
        (
            'stats --r /path/to/reports --status skipped',
            triton_utils.Config(
                action='tests_stats',
                reports='/path/to/reports',
                status_filter=['skipped'],
            )
        ),
        (
            'compare --r /path/to/reports --r2 /path/to/reports2',
            triton_utils.Config(
                action='compare_reports',
                reports='/path/to/reports',
                reports_2='/path/to/reports2'
            )
        ),
    ]
)
def test_args(args: str, config: triton_utils.Config):
    triton_utils.Config.from_args(args)
    assert triton_utils.Config.from_args(args) == config


@pytest.mark.parametrize(
    ('args', 'results_row','results', 'grouping_level'),
    [
        (
            {'exclude_subdir_patterns': [re.compile(r'subdir2')], 'include_subdir_patterns': [re.compile(r'^.*$')]},
            'language',
            {'passed': [1, 0, -1], 'skipped': [1, 2, 1], 'xfailed': [1, 1, 0], 'failed': [2, 2, 0]},
            'testsuite'
        ),
        (
            {'exclude_subdir_patterns': [re.compile(r'subdir2')], 'include_subdir_patterns': [re.compile(r'^.*$')]},
            'language::test_core.test_reduce_layouts',
            {'passed': [1, 0, -1]},
            'test'
        ),
    ]
)
def test_compare_reports(  # pylint: disable=R0913, R0914, R0917
    tmp_path,
    args: dict[str, Any],
    results_row,
    results,
    grouping_level,
):
    for report_name, doc_str in TESTS_WITH_MULTIPLE_RESULTS.items():
        report_name_parts = report_name.split('/')
        subdir_path = tmp_path / report_name_parts[0]
        subdir_path.mkdir(parents=True, exist_ok=True)
        report_file = subdir_path / report_name_parts[1]
        report_file.write_text(doc_str, encoding='utf-8')

    config = dataclasses.replace(
        triton_utils.Config(
            action='compare_reports',
            reports=str(tmp_path / 'test-report1'),
            reports_2=str(tmp_path / 'test-report2'),
            _report_grouping_level=triton_utils.TestGroupingLevel(grouping_level).value,
            **args
        ),
    )

    comparision = triton_utils.run(config)
    for key, result in results.items():
        assert comparision.loc[results_row, (key, 'r1')] == result[0]
        assert comparision.loc[results_row, (key, 'r2')] == result[1]
        assert comparision.loc[results_row, (key, 'Δ')] == result[2]


TESTS_WITH_FLAKY_RESULTS = {
    'test-report1/language.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="0" failures="1" skipped="0" tests="2" time="86.246" timestamp="2024-04-05T11:03:23.033702" hostname="hostname">
        <testcase classname="python.test.unit.language.test_core" name="test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]" time="0.001" />
        <testcase classname="python.test.unit.language.test_matmul" name="test_blocked_scale_mxfp[False-4-128-128-256-1024-512-512]" time="86.245">
            <failure message="AssertionError: Tensor-likes are not close! ...">
                M = 1024, N = 512, K = 512, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 256
                language/test_matmul.py:859: AssertionError
            </failure>
        </testcase>
    </testsuite>
</testsuites>
''', 'test-report2/language.xml':
    '''<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="1" failures="0" skipped="0" tests="2" time="86.246" timestamp="2024-04-05T11:03:23.033702" hostname="hostname">
        <testcase classname="python.test.unit.language.test_core" name="test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]" time="0.001">
            <error message="failed on setup with &quot;worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'&quot;">
                worker 'gw14' crashed while running 'test/unit/language/test_core.py::test_dot[1-64-128-128-2-False-False-none-tf32-float32-float32]'
            </error>
        </testcase>
        <testcase classname="python.test.unit.language.test_matmul" name="test_blocked_scale_mxfp[False-4-128-128-256-1024-512-512]" time="86.245" />
    </testsuite>
</testsuites>
'''
}

@pytest.mark.parametrize(
    ('reports', 'is_flaky', 'passed', 'skipped', 'xfailed', 'failed'),
    [
        (TESTS_WITH_FLAKY_RESULTS, True, 2, 0, 0, 0,),
        ({next(iter(TESTS_WITH_FLAKY_RESULTS.items()))[0]: next(iter(TESTS_WITH_FLAKY_RESULTS.items()))[1]}, False, 1, 0, 0, 1,),
    ]
)
def test_flaky_tests_detection(  # pylint: disable=R0913, R0914, R0917
    tmp_path, capsys, reports, is_flaky, passed, skipped, xfailed, failed,
):
    for report_name, doc_str in reports.items():
        report_name_parts = report_name.split('/')
        subdir_path = tmp_path / report_name_parts[0]
        subdir_path.mkdir(parents=True, exist_ok=True)
        report_file = subdir_path / report_name_parts[1]
        report_file.write_text(doc_str, encoding='utf-8')


    config = triton_utils.Config(
        action='pass_rate',
        reports=str(tmp_path),
        merge_test_results=True,
    )

    out, _ = triton_utils.PassRateActionRunner(config)()

    warnings_out, _ = capsys.readouterr()

    pass_rate_dict = extract_pass_rate_dict(out)
    assert pass_rate_dict['passed'] == passed
    assert pass_rate_dict['skipped'] == skipped
    assert pass_rate_dict['xfailed'] == xfailed
    assert pass_rate_dict['failed'] == failed

    assert is_flaky and '[WARNING] Flaky test detected:' in warnings_out or not is_flaky

# yapf: enable
