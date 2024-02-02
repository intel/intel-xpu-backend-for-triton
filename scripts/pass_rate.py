"""Calculates and reports pass rate for unit tests."""

import argparse
import dataclasses
import pathlib
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
        help='directory with testing reports (JUnit XML), default: %(default)s',
    )
    return argument_parser


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
        f' pass rate: {round(100 * stats.passed / stats.total, 2)}%,'
        f' pass rate (w/o xfailed): {round(100 * stats.passed / (stats.total - stats.xfailed), 2)}%'
    )  # yapf: disable


def main():
    """Main."""
    args = create_argument_parser().parse_args()
    stats = parse_reports(pathlib.Path(args.reports))
    for item in stats:
        print_stats(item)
    print_stats(overall_stats(stats))


if __name__ == '__main__':
    main()
