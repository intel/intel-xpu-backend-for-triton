"""Calculates and reports pass rate for unit tests."""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
import sys
from typing import Dict, List, Optional

from defusedxml.ElementTree import parse


@dataclasses.dataclass
class ReportStats:
    """Report stats."""
    name: str = ''
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    fixme: int = 0
    total: int = 0

    @property
    def pass_rate(self):
        """Pass rate."""
        if self.total == 0:
            return 0.0
        return round(100 * self.passed / self.total, 2)

    @property
    def pass_rate_without_xfailed(self):
        """Pass rate without xfailed."""
        if self.total - self.xfailed == 0:
            return 0.0
        return round(100 * self.passed / (self.total - self.xfailed), 2)


@dataclasses.dataclass
class TestWarning:
    """Test warning."""
    location: str
    message: str


def create_argument_parser() -> argparse.ArgumentParser:
    """Creates ArgumentParser."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--reports',
        default='.',
        type=str,
        help='directory with reports (JUnit XML), default: %(default)s',
    )
    argument_parser.add_argument(
        '--json',
        action='store_true',
        help='print stats in JSON',
    )
    argument_parser.add_argument(
        '--suite',
        type=str,
        default='all',
        help='name of the test suite, default: %(default)s',
    )
    argument_parser.add_argument(
        '--skip-list',
        type=str,
        help='an exclude list dir used in pass rate calculation',
    )
    return argument_parser


def get_deselected(report_path: pathlib.Path, skiplist_dir: pathlib.Path) -> int:
    """Calculates deselected (via skiplist) tests."""
    skiplist_path = skiplist_dir / f'{report_path.stem}.txt'
    if not skiplist_path.exists():
        return 0
    with skiplist_path.open('r') as f:
        count = 0
        for line in f.readlines():
            # `strip` allows to skip lines with only '\n' character
            line = line.strip()
            # skip empty lines and comments
            if line and not line.startswith('#'):
                count += 1
        return count


def get_warnings(reports_path: pathlib.Path, suite: str) -> List[TestWarning]:
    """Returns a list of warnings for the specified suite."""
    path = reports_path / f'{suite}-warnings.json'
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as warnings_file:
        warnings_data = json.load(warnings_file)
    return [TestWarning(location=next(iter(w.keys())), message=next(iter(w.values()))) for w in warnings_data]


def get_missing_tests(warnings: List[TestWarning]) -> List[str]:
    """Searches warnings for PytestSelectWarning and returns a list of missing tests."""
    tests = set()
    for warning in warnings:
        if 'PytestSelectWarning: pytest-select: Not all deselected' not in warning.message:
            continue
        for line in warning.message.splitlines():
            if line.startswith('  - '):
                tests.add(line.removeprefix('  - '))
    return sorted(list(tests))


def get_all_missing_tests(reports_path: pathlib.Path) -> Dict[str, List[str]]:
    """Returns missing tests for all suites."""
    all_missing_tests = {}
    for report in reports_path.glob('*.xml'):
        suite = report.stem
        warnings = get_warnings(reports_path, suite)
        missing_tests = get_missing_tests(warnings)
        if missing_tests:
            all_missing_tests[suite] = missing_tests
    return all_missing_tests


def parse_report(report_path: pathlib.Path, skiplist_dir: Optional[pathlib.Path]) -> ReportStats:
    """Parses the specified report."""
    stats = ReportStats(name=report_path.stem)
    root = parse(report_path).getroot()
    for testsuite in root:
        testsuite_fixme_tests = set()
        stats.total += int(testsuite.get('tests'))
        for skipped in testsuite.iter('skipped'):
            if skipped.get('type') == 'pytest.skip':
                stats.skipped += 1
            elif skipped.get('type') == 'pytest.xfail':
                stats.xfailed += 1
        for _ in testsuite.iter('failure'):
            stats.failed += 1
        for _ in testsuite.iter('error'):
            stats.failed += 1
        for warning in get_warnings(report_path.parent, report_path.stem):
            if 'FIXME' in warning.message:
                testsuite_fixme_tests.add(warning.location)
        stats.fixme += len(testsuite_fixme_tests)

    test_unskip = os.getenv('TEST_UNSKIP', 'false')
    if test_unskip not in ('true', 'false'):
        raise ValueError('Error: please set TEST_UNSKIP true or false')
    if skiplist_dir and test_unskip == 'false':
        deselected = get_deselected(report_path, skiplist_dir)
        stats.skipped += deselected
        stats.total += deselected
    stats.passed = stats.total - stats.failed - stats.skipped - stats.xfailed
    return stats


def overall_stats(stats: List[ReportStats]) -> ReportStats:
    """Returns overall stats."""
    overall = ReportStats(name='all')
    for item in stats:
        overall.passed += item.passed
        overall.failed += item.failed
        overall.skipped += item.skipped
        overall.xfailed += item.xfailed
        overall.total += item.total
        overall.fixme += item.fixme
    return overall


def find_stats(stats: List[ReportStats], name: str) -> ReportStats:
    """Finds stats by name."""
    for item in stats:
        if item.name == name:
            return item
    raise ValueError(f'{name} not found')


def parse_junit_reports(args: argparse.Namespace) -> List[ReportStats]:
    """Parses junit report in the specified directory."""
    reports_path = pathlib.Path(args.reports)
    return [parse_report(report, args.skiplist_dir) for report in reports_path.glob('*.xml')]


def parse_tutorials_reports(args: argparse.Namespace) -> List[ReportStats]:
    """Parses tutorials reports in the specified directory."""
    reports_path = pathlib.Path(args.reports)
    stats = ReportStats(name='tutorials')
    for report in reports_path.glob('tutorial-*.txt'):
        result = report.read_text().strip()
        stats.total += 1
        if result == 'PASS':
            stats.passed += 1
        elif result == 'SKIP':
            stats.skipped += 1
        elif result == 'FAIL':
            stats.failed += 1
    return [stats]


def parse_reports(args: argparse.Namespace) -> List[ReportStats]:
    """Parses all report in the specified directory."""
    return parse_junit_reports(args) + parse_tutorials_reports(args)


def print_text_stats(stats: ReportStats):
    """Prints report stats."""
    print(
        f'{stats.name}:'
        f' passed: {stats.passed},'
        f' failed: {stats.failed},'
        f' skipped: {stats.skipped},'
        f' xfailed: {stats.xfailed},'
        f' total: {stats.total},'
        f' fixme: {stats.fixme},'
        f' pass rate (w/o xfailed): {stats.pass_rate_without_xfailed}%'
    )  # yapf: disable


def print_json_stats(stats: ReportStats):
    """Print JSON stats."""
    data = {
        'ts': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'os': platform.system(),
        'git_ref': os.getenv('GITHUB_REF_NAME', ''),
        'git_sha': os.getenv('GITHUB_SHA', ''),
        'libigc1_version': os.getenv('LIBIGC1_VERSION', ''),
        'level_zero_version': os.getenv('LEVEL_ZERO_VERSION', ''),
        'agama_version': os.getenv('AGAMA_VERSION', ''),
        'gpu_device': os.getenv('GPU_DEVICE', ''),
        'python_version': platform.python_version(),
        'pytorch_version': os.getenv('PYTORCH_VERSION', ''),
        'testsuite': stats.name,
        'passed': stats.passed,
        'failed': stats.failed,
        'skipped': stats.skipped,
        'xfailed': stats.xfailed,
        'total': stats.total,
        'fixme': stats.fixme,
        'pass_rate_1': stats.pass_rate,
        'pass_rate_2': stats.pass_rate_without_xfailed,
    }  # yapf: disable
    print(json.dumps(data, indent=2))


def main():
    """Main."""
    args = create_argument_parser().parse_args()
    args.report_path = pathlib.Path(args.reports)
    if args.skip_list:
        args.skiplist_dir = pathlib.Path(args.skip_list)
    else:
        args.skiplist_dir = None

    missing_tests = get_all_missing_tests(args.report_path)
    if missing_tests:
        for suite, tests in missing_tests.items():
            print(f'# Missing tests in {suite}:')
            for test in tests:
                print(test)
        sys.exit(1)

    stats = parse_reports(args)

    if args.suite == 'all':
        summary = overall_stats(stats)
    else:
        summary = find_stats(stats, args.suite)

    if args.json:
        print_json_stats(summary)
    else:
        if args.suite == 'all':
            for item in sorted(stats, key=lambda x: x.name):
                print_text_stats(item)
        print_text_stats(summary)


if __name__ == '__main__':
    main()
