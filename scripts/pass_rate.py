"""Calculates and reports pass rate for unit tests."""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
from typing import List

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
    return argument_parser


def get_deselected(report_path: pathlib.Path) -> int:
    """Calculates deselected (via skiplist) tests."""
    skiplist_dir = os.getenv('TRITON_TEST_SKIPLIST_DIR', 'scripts/skiplist/default')
    skiplist_path = pathlib.Path(skiplist_dir) / f'{report_path.stem}.txt'
    if not skiplist_path.exists():
        return 0
    with skiplist_path.open('r') as f:
        # skip empty lines and comments
        return len([line for line in f.readlines() if line and not line.startswith('#')])


def parse_report(report_path: pathlib.Path) -> ReportStats:
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
        try:
            warnings_file_name = f'{report_path.parent}/{report_path.stem}-warnings.json'
            with open(warnings_file_name, encoding='utf-8') as testsuite_warnings_file:
                testsuite_warnings = json.load(testsuite_warnings_file)
                for w in testsuite_warnings:
                    if 'FIXME' in list(w.values())[0]:
                        testsuite_fixme_tests.add(list(w.keys())[0])
        except FileNotFoundError:
            pass
        stats.fixme += len(testsuite_fixme_tests)

    test_unskip = os.getenv('TEST_UNSKIP')
    if test_unskip not in ('true', 'false'):
        raise ValueError('Error: please set TEST_UNSKIP true or false')
    if test_unskip == 'false':
        deselected = get_deselected(report_path)
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


def parse_junit_reports(reports_path: pathlib.Path) -> List[ReportStats]:
    """Parses junit report in the specified directory."""
    return [parse_report(report) for report in reports_path.glob('*.xml')]


def parse_tutorials_reports(reports_path: pathlib.Path) -> List[ReportStats]:
    """Parses tutorials reports in the specified directory."""
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


def parse_reports(reports_path: pathlib.Path) -> List[ReportStats]:
    """Parses all report in the specified directory."""
    return parse_junit_reports(reports_path) + parse_tutorials_reports(reports_path)


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
    stats = parse_reports(pathlib.Path(args.reports))

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
