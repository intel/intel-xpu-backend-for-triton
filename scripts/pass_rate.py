"""Calculates and reports pass rate for unit tests."""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
import xml.etree.ElementTree as ET
from typing import List


@dataclasses.dataclass
class ReportStats:
    """Report stats."""
    name: str = ''
    passed: int = 0
    skipped: int = 0
    xfailed: int = 0
    total: int = 0


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
    return argument_parser


def get_deselected(report_path: pathlib.Path) -> int:
    """Calculates deselected (via skiplist) tests."""
    skiplist_dir = os.getenv('TRITON_TEST_SKIPLIST_DIR', 'scripts/skiplist/default')
    skiplist_path = pathlib.Path(skiplist_dir) / f'{report_path.stem}.txt'
    if not skiplist_path.exists():
        return 0
    with skiplist_path.open('r') as f:
        # Return the number of lines except comments
        return len([line for line in f.readlines() if not line.startswith('#')])


def parse_report(report_path: pathlib.Path) -> ReportStats:
    """Parses the specified report."""
    stats = ReportStats(name=report_path.stem)
    root = ET.parse(report_path).getroot()
    for testsuite in root:
        stats.total += int(testsuite.get('tests'))
        for skipped in testsuite.iter('skipped'):
            if skipped.get('type') == 'pytest.skip':
                stats.skipped += 1
            elif skipped.get('type') == 'pytest.xfail':
                stats.xfailed += 1
    deselected = get_deselected(report_path)
    stats.skipped += deselected
    stats.total += deselected
    stats.passed = stats.total - stats.skipped - stats.xfailed
    return stats


def overall_stats(stats: List[ReportStats]) -> ReportStats:
    """Returns overall stats."""
    overall = ReportStats(name='all')
    for item in stats:
        overall.passed += item.passed
        overall.skipped += item.skipped
        overall.xfailed += item.xfailed
        overall.total += item.total
    return overall


def parse_reports(reports_path: pathlib.Path) -> List[ReportStats]:
    """Parses all report in the specified directory."""
    return [parse_report(report) for report in reports_path.glob('*.xml')]


def print_stats(stats: ReportStats):
    """Prints report stats."""
    print(
        f'{stats.name}:'
        f' passed: {stats.passed},'
        f' skipped: {stats.skipped},'
        f' xfailed: {stats.xfailed},'
        f' total: {stats.total},'
        f' pass rate (w/o xfailed): {round(100 * stats.passed / (stats.total - stats.xfailed), 2)}%'
    )  # yapf: disable


def print_text_stats(stats: List[ReportStats]):
    """Prints human readable stats."""
    for item in stats:
        print_stats(item)
    print_stats(overall_stats(stats))


def print_json_stats(stats: List[ReportStats]):
    """Print JSON stats."""
    overall = overall_stats(stats)
    data = {
        'ts': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'git_ref': os.getenv('GITHUB_REF_NAME', ''),
        'git_sha': os.getenv('GITHUB_SHA', ''),
        'python_version': platform.python_version(),
        'testsuite': overall.name,
        'passed': overall.passed,
        'skipped': overall.skipped,
        'xfailed': overall.xfailed,
        'total': overall.total,
        'pass_rate_1': round(100 * overall.passed / overall.total, 2),
        'pass_rate_2': round(100 * overall.passed / (overall.total - overall.xfailed), 2)
    }  # yapf: disable
    print(json.dumps(data, indent=2))


def main():
    """Main."""
    args = create_argument_parser().parse_args()
    stats = parse_reports(pathlib.Path(args.reports))
    if args.json:
        print_json_stats(stats)
    else:
        print_text_stats(stats)


if __name__ == '__main__':
    main()
