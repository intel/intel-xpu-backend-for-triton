"""Calculates and reports pass rate for unit tests."""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
import xml.etree.ElementTree as et

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
    return argument_parser


def get_warnings(reports_path: pathlib.Path, suite: str) -> List[TestWarning]:
    """Returns a list of warnings for the specified suite."""
    path = reports_path / f'{suite}-warnings.json'
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as warnings_file:
        warnings_data = json.load(warnings_file)
    return [TestWarning(location=next(iter(w.keys())), message=next(iter(w.values()))) for w in warnings_data]


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
        for warning in get_warnings(report_path.parent, report_path.stem):
            if 'FIXME' in warning.message:
                testsuite_fixme_tests.add(warning.location)
        stats.fixme += len(testsuite_fixme_tests)

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


# pylint: disable=too-many-locals
def generate_junit_report(reports_path: pathlib.Path):
    """Parses info files for tutorials and generates JUnit report.
    The script `run_tutorial.py` generates `tutorial-*.json` files in the reports directory.
    This function loads them and generates `tutorials.xml` file (JUnit XML report) in the same
    directory.
    """
    testsuites = et.Element('testsuites')
    testsuite = et.SubElement(testsuites, 'testsuite', name='tutorials')

    total_tests, total_errors, total_failures, total_skipped = 0, 0, 0, 0
    total_time = 0.0

    for item in reports_path.glob('tutorial-*.json'):
        data = json.loads(item.read_text())
        name, result, time = data['name'], data['result'], data.get('time', 0)
        testcase = et.SubElement(testsuite, 'testcase', name=name)
        if result == 'PASS':
            testcase.set('time', str(time))
        elif result == 'SKIP':
            total_skipped += 1
            et.SubElement(testcase, 'skipped', type='pytest.skip')
        elif result == 'FAIL':
            total_failures += 1
            et.SubElement(testcase, 'failure', message=data.get('message', ''))
        else:
            continue
        total_tests += 1
        total_time += time

    testsuite.set('tests', str(total_tests))
    testsuite.set('errors', str(total_errors))
    testsuite.set('failures', str(total_failures))
    testsuite.set('skipped', str(total_skipped))
    testsuite.set('time', str(total_time))

    report_path = reports_path / 'tutorials.xml'
    with report_path.open('wb') as f:
        tree = et.ElementTree(testsuites)
        tree.write(f, encoding='UTF-8', xml_declaration=True)


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

    reports_path = pathlib.Path(args.reports)
    generate_junit_report(reports_path)
    stats = parse_junit_reports(reports_path)

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
