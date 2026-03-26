from __future__ import annotations

from typing import ClassVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import sys
import shlex
import argparse
from pathlib import Path

import re

from typing_extensions import Any, Self

import pandas as pd

from .pass_rate_utils import Test, TestReport, TestGroupingLevel, CompareScope, SortByStats, SortByCompare
from .gh_utils import GHANightlyTestReportProcessor, GHABuildTestReportProcessor, GHAWheelDownloader
from .pattern_matcher import PatternMatcher


class AppendOrReplace(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        self._default = kwargs.get("default")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is None or current == self._default:
            setattr(namespace, self.dest, [values])
        else:
            current.append(values)


@dataclass
class Config:  # pylint: disable=R0902
    action: str | None = None

    reports: str | None = None
    reports_2: str | None = None

    status_filter: list[str] = field(default_factory=lambda: ["passed", "skipped", "failed", "xfailed"])
    suite: str | None = None
    ignore_testsuite_filter: list[str] = field(default_factory=lambda: [])
    testname_filter: str | None = None
    include_subdir_patterns: list[re.Pattern[str]] = field(
        default_factory=lambda: [re.compile(r"^test-report(?!.*lts$).*")], )
    exclude_subdir_patterns: list[re.Pattern[str]] = field(
        default_factory=lambda: [re.compile(r"^(?!)$")]  # Match nothing by default
    )
    merge_test_results: bool = False

    error_on_failures: bool = False
    tests_with_multiple_testsuites: bool = False

    _report_grouping_level: str = TestGroupingLevel.TEST.value
    list_test_instances: bool = False
    list_failure_reasons: bool = False
    pretty_print: bool = False
    long_names: bool = False
    sort_by: str = "name"
    _compare_scope: str = CompareScope.ANY.value
    omit_testsuite_name: bool = False
    omit_test_module_name: bool = False
    omit_test_class_name: bool = False

    repo: str = "intel/intel-xpu-backend-for-triton"
    branch: str = "main"
    reports_dir: str | None = None

    _download_dir: str | None = None
    artifact_pattern: str | None = None

    nightly_run_id: str | None = None
    latest_nightly_gh_run: bool = False
    gh_run_id: str | None = None

    export_format: str = "csv"
    file_name: str = "test_report.csv"

    save_to_json: str | None = None
    pass_rate_level: str = "all"

    # download_wheels fields
    wheel_set: list[str] = field(default_factory=list)
    python_version: str | None = None
    download_for_all_pythons: bool = False
    latest_wf_run: str | None = None
    latest_wf_run_pattern: str | None = None

    @property
    def report_grouping_level(self) -> TestGroupingLevel:
        return TestGroupingLevel(self._report_grouping_level)

    @property
    def compare_scope(self) -> CompareScope:
        return CompareScope(self._compare_scope)

    @property
    def download_dir(self) -> Path:
        return Path(str(self._download_dir))

    @classmethod
    def add_reports_source_args(cls, parser: argparse.ArgumentParser, compare_reports: bool = False):
        parser.add_argument(
            "--reports",
            "--r",
            type=str,
            required=True,
            help="Path to the reports folder",
        )
        if compare_reports:
            parser.add_argument(
                "--reports-2",
                "--r2",
                type=str,
                required=True,
                help="Path to the second reports folder",
            )

    @classmethod
    def add_filter_args(cls, parser: argparse.ArgumentParser):
        # Common filters for all actions
        all_statuses = cls().status_filter
        parser.add_argument(
            "--status",
            action=AppendOrReplace,
            choices=all_statuses,
            dest="status_filter",
            required=False,
            default=all_statuses,
            help=f"Filter by result statuses - {all_statuses}, default value is all",
        )
        parser.add_argument(
            "--suite",
            type=str,
            required=False,
            help="Filter by testsuite name",
        )
        parser.add_argument(
            "--test",
            type=str,
            required=False,
            dest="testname_filter",
            help="Filter by test name",
        )
        parser.add_argument(
            "--ignore-testsuite",
            "--ignore-suite",
            action=AppendOrReplace,
            required=False,
            dest="ignore_testsuite_filter",
            default=cls().ignore_testsuite_filter,
            help="Ignore test suite by name",
        )
        include_pattern_default = [
            "--include-subdir-pattern " + '"' + pattern.pattern + '"' for pattern in cls().include_subdir_patterns
        ]
        parser.add_argument(
            "--include-subdir-pattern",
            "--include-dir",
            action=AppendOrReplace,
            dest="include_subdir_patterns",
            default=cls().include_subdir_patterns,
            required=False,
            help=
            (f"Include patterns for report subdir or artifact name, default value is `{' '.join(include_pattern_default)}`."
             f" If the report subdir or artifact name matches include and exclude patterns, exclude pattern will have a priority"
             ),
        )
        exclude_pattern_default = [
            "--exclude-subdir-pattern " + '"' + pattern.pattern + '"' for pattern in cls().exclude_subdir_patterns
        ]
        parser.add_argument(
            "--exclude-subdir-pattern",
            "--exclude-dir",
            action=AppendOrReplace,
            dest="exclude_subdir_patterns",
            default=cls().exclude_subdir_patterns,
            required=False,
            help=
            f"Exclude patterns for report subdir or artifact name, default value is `{' '.join(exclude_pattern_default)}`.",
        )
        parser.add_argument(
            "--merge-test-results",
            "--merge",
            action="store_true",
            dest="merge_test_results",
            required=False,
        )
        parser.add_argument(
            "--long-names",
            action="store_true",
            required=False,
            help=
            "Display full test names in pretty-printed tables. May produce wide output; consider redirecting to a file for easier reading.",
        )

    _ALIASES: ClassVar[dict[str, str]] = {
        "stats": "tests_stats",
        "compare": "compare_reports",
        "export": "export_to",
        "download": "download_reports",
        "wheels": "download_wheels",
    }

    _CANONICAL_NAMES: ClassVar[dict[str, str]] = {v: k for k, v in _ALIASES.items()}

    @classmethod
    def args_to_canonical(cls, args: argparse.Namespace) -> argparse.Namespace:
        if hasattr(args, "action") and args.action in cls._ALIASES:
            args.action = cls._ALIASES[args.action]
        return args

    @classmethod
    def _add_parser(cls, subparsers, name: str, help_str: str) -> argparse.ArgumentParser:
        if name in cls._ALIASES.values():
            return subparsers.add_parser(name, aliases=[cls._CANONICAL_NAMES[name]], help=help_str)
        return subparsers.add_parser(name, help=help_str)

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:  # pylint: disable=R0915
        parser = argparse.ArgumentParser(add_help=False)

        subparsers = parser.add_subparsers(dest="action", required=True)

        pass_rate_parser = cls._add_parser(
            subparsers,
            "pass_rate",
            help_str="Calculate pass rate",
        )
        cls.add_reports_source_args(pass_rate_parser)
        cls.add_filter_args(pass_rate_parser)
        pass_rate_parser.add_argument(
            "--tests-with-multiple-testsuites",
            "--multiple-suites",
            "--mutiple",
            action="store_true",
            help="Allow tests to be present im multiple testsuites",
        )
        pass_rate_parser.add_argument(
            "--error-on-failures",
            action="store_true",
            help="Fail with error code if there are any test failures",
        )
        pass_rate_parser.add_argument(
            "--save-to-json",
            "--json",
            type=str,
            required=False,
            help="Json file to save pass rate summary.",
        )
        pass_rate_parser.add_argument(
            "--level",
            choices=["all", "testsuite"],
            default="all",
            dest="pass_rate_level",
            help="Grouping level for pass rate report (default: all)",
        )

        test_stats_parser = cls._add_parser(
            subparsers,
            "tests_stats",
            help_str="Print tests summary",
        )
        cls.add_reports_source_args(test_stats_parser)
        cls.add_filter_args(test_stats_parser)
        test_stats_parser.add_argument(
            "--list-test-instances",
            "--list",
            action="store_true",
            help="List test instances",
        )
        test_stats_parser.add_argument(
            "--sort-by",
            default=cls().sort_by,
            type=str,
            choices=[s.value for s in SortByStats],
            help="Sort by column name",
        )
        test_stats_parser.add_argument(
            "--tests-with-multiple-testsuites",
            "--multiple-suites",
            "--mutiple",
            action="store_true",
            help="Allow tests to be present im multiple testsuites",
        )
        test_stats_parser.add_argument(
            "--report-grouping-level",
            "--level",
            choices=[level.value for level in TestGroupingLevel],
            default=cls()._report_grouping_level,
            dest="_report_grouping_level",
            help="Grouping level for the report",
        )
        test_stats_parser.add_argument(
            "--list-failure-reasons",
            "--failures",
            action="store_true",
            help="List failure reasons for failed tests",
        )
        test_stats_parser.add_argument(
            "--pretty-print",
            "--pretty",
            action="store_true",
            required=False,
            help="Pretty print stats",
        )

        compare_stats_parser = cls._add_parser(
            subparsers,
            "compare_reports",
            help_str="Compare reports",
        )
        cls.add_reports_source_args(compare_stats_parser, compare_reports=True)
        cls.add_filter_args(compare_stats_parser)
        compare_stats_parser.add_argument(
            "--tests-with-multiple-testsuites",
            "--multiple-suites",
            "--mutiple",
            action="store_true",
            help="Allow tests to be present im multiple testsuites",
        )
        compare_stats_parser.add_argument(
            "--report-grouping-level",
            "--level",
            choices=[level.value for level in TestGroupingLevel],
            default=cls()._report_grouping_level,
            dest="_report_grouping_level",
            help="Grouping level for the report",
        )
        compare_stats_parser.add_argument(
            "--sort-by",
            default=cls().sort_by,
            type=str,
            choices=([s.value for s in SortByCompare] +
                     [s.value.replace(".Δ", ".delta") for s in SortByCompare if ".Δ" in s.value] +
                     [s.value.replace(".%Δ", ".%delta") for s in SortByCompare if ".%Δ" in s.value]),
            help="Sort by column in <metric>.<source> format (e.g., passed.r1, time.delta)",
        )
        compare_stats_parser.add_argument(
            "--pretty-print",
            "--pretty",
            action="store_true",
            required=False,
            help="Pretty print (accepted for consistency, compare always outputs a table)",
        )
        compare_stats_parser.add_argument(
            "--compare-scope",
            choices=[scope.value for scope in CompareScope],
            default=cls()._compare_scope,
            dest="_compare_scope",
            help="Filter: any (all), r1-only (in r1 not r2), r2-only (in r2 not r1), both",
        )
        compare_stats_parser.add_argument(
            "--omit-testsuite-name",
            action="store_true",
            required=False,
            help="Omit testsuite name prefix from displayed test names",
        )
        compare_stats_parser.add_argument(
            "--omit-test-module-name",
            action="store_true",
            required=False,
            help="Omit test module path and name from displayed test names",
        )
        compare_stats_parser.add_argument(
            "--omit-test-class-name",
            action="store_true",
            required=False,
            help="Omit test class name from displayed test names",
        )

        convert_to_parser = cls._add_parser(
            subparsers,
            "export_to",
            help_str="Convert report to another format",
        )
        cls.add_reports_source_args(convert_to_parser)
        cls.add_filter_args(convert_to_parser)
        convert_to_parser.add_argument(
            "--tests-with-multiple-testsuites",
            "--multiple-suites",
            "--mutiple",
            action="store_true",
            help="Allow tests to be present im multiple testsuites",
        )
        convert_to_parser.add_argument(
            "--format",
            choices=["csv"],
            dest="export_format",
            required=True,
            help="Export format",
        )
        convert_to_parser.add_argument(
            "--file-name",
            required=True,
            help="Export filename",
        )

        nightly_parser = cls._add_parser(
            subparsers,
            "download_reports",
            help_str="Download reports",
        )
        nightly_parser.add_argument(
            "--download-dir",
            "-D",
            type=str,
            required=True,
            dest="_download_dir",
            help="Directory to download reports",
        )
        nightly_parser.add_argument(
            "--repo",
            "-R",
            type=str,
            required=False,
            default="intel/intel-xpu-backend-for-triton",
            help="Default repo",
        )
        nightly_parser.add_argument(
            "--branch",
            "-B",
            type=str,
            required=False,
            default="main",
            help="Default repo",
        )
        nightly_parser.add_argument(
            "--artifact-pattern",
            type=str,
            required=False,
            default=None,
            help="Glob pattern to filter artifact names (e.g. 'test-reports-xe2-*')",
        )
        source_run_group = nightly_parser.add_mutually_exclusive_group(required=True)
        source_run_group.add_argument(
            "--nightly-run-id",
            "--nightly",
            type=str,
            required=False,
            default="",
            help="GH nightly run id",
        )
        source_run_group.add_argument(
            "--latest-nightly-gh-run",
            "--latest-nightly",
            action="store_true",
            help="Latest-nightly GH run",
        )
        source_run_group.add_argument(
            "--gh-run-id",
            "--run",
            type=str,
            required=False,
            default="",
            help="GH run id",
        )
        wheels_parser = cls._add_parser(
            subparsers,
            "download_wheels",
            help_str="Download wheel artifacts from CI",
        )
        wheels_parser.add_argument(
            "--download-dir",
            "-D",
            type=str,
            required=True,
            dest="_download_dir",
            help="Directory to download wheels to",
        )
        wheels_parser.add_argument(
            "--repo",
            "-R",
            type=str,
            required=False,
            default="intel/intel-xpu-backend-for-triton",
            help="GitHub repository (default: intel/intel-xpu-backend-for-triton)",
        )
        wheels_parser.add_argument(
            "--branch",
            "-B",
            type=str,
            required=False,
            default="main",
            help="Branch to search for successful runs (default: main)",
        )
        wheels_parser.add_argument(
            "--wheel-set",
            "--ws",
            action="append",
            dest="wheel_set",
            default=[],
            choices=list(GHAWheelDownloader.WHEEL_SETS.keys()),
            help=("Filter by predefined wheel set (repeatable). "
                  "Presets: torch (torch, torchvision, torchaudio, timm), "
                  "triton (triton), bench (triton_kernels_benchmark), pti (intel_pti)"),
        )
        wheels_parser.add_argument(
            "--artifact-pattern",
            type=str,
            required=False,
            default=None,
            help="fnmatch pattern to filter wheel filenames (power-user)",
        )
        wheels_parser.add_argument(
            "--python-version",
            type=str,
            required=False,
            default=None,
            help="Python version for artifact matching (default: current python)",
        )
        wheels_parser.add_argument(
            "--download-for-all-pythons",
            action="store_true",
            help="Disable Python version filtering — download wheels for all Python versions",
        )
        wheels_run_group = wheels_parser.add_mutually_exclusive_group()
        wheels_run_group.add_argument(
            "--gh-run-id",
            "--run",
            type=str,
            required=False,
            default=None,
            help="Specific GH Actions run ID to download from",
        )
        wheels_run_group.add_argument(
            "--latest-wf-run",
            type=str,
            required=False,
            default=None,
            help=("Preset name for workflow to find latest successful run. "
                  "Presets: nightly (default), benchmarks, build-test, wheels, wheels-triton, wheels-pytorch"),
        )
        wheels_run_group.add_argument(
            "--latest-wf-run-pattern",
            type=str,
            required=False,
            default=None,
            help="fnmatch pattern to match workflow path (e.g., 'build-benchmarks-*.yml')",
        )

        return parser

    @classmethod
    def from_args(cls, arg_string: str | None = None) -> Self:
        parser = cls.build_parser()
        if arg_string is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(arg_string))
        args = cls.args_to_canonical(args)
        return cls(**vars(args))


@dataclass
class ActionRunner(ABC):
    config: Config


@dataclass
class ReportActionRunner(ActionRunner):
    reports: list[TestReport] = field(init=False)
    base_report: TestReport = field(init=False)

    def __post_init__(self):
        config = self.config

        if not config.reports:
            raise ValueError("Path to the reports folder should be provided")

        reports_paths: list[str] = [rep_path for rep_path in [config.reports, config.reports_2] if rep_path is not None]
        test_reports: list[TestReport] = []

        for rep_path in reports_paths:
            test_report = TestReport.from_reports_folder(
                rep_path,
                tests_with_multiple_testsuites=config.tests_with_multiple_testsuites,
                ignore_testsuites_filter=config.ignore_testsuite_filter,
                pattern_matcher=PatternMatcher(
                    include_patterns=config.include_subdir_patterns,
                    exclude_patterns=config.exclude_subdir_patterns,
                ),
            )
            test_report = test_report.filter_test_instances(
                status_filter=config.status_filter,
                testsuite=config.suite,
                testname_wo_variant=config.testname_filter,
                ignore_testsuites=config.ignore_testsuite_filter,
            )
            if test_report.check_for_multiple_test_results():
                if not config.merge_test_results:
                    print(
                        "[WARNING] Multiple test results for the same test case have been found. Use --merge-test-results to filter the results by the best outcome only."
                    )
                else:
                    test_report = test_report.merge_test_results()
            test_reports.append(test_report)
        self.reports = test_reports
        self.base_report = test_reports[0]

    @property
    def summary(self) -> str:
        return self.base_report.get_pass_rate_summary()

    @property
    def tests(self) -> list[Test]:
        return list(self.base_report.tests.values())

    @property
    def summary_detailed(self) -> str:
        config = self.config
        if (config.report_grouping_level != TestGroupingLevel.TEST and config.list_test_instances):
            raise NotImplementedError("--list-test-instances is only supported with --report-grouping-level test")
        return self.base_report.get_summary(
            grouping_level=config.report_grouping_level,
            list_test_instances=config.list_test_instances,
            list_failure_reasons=config.list_failure_reasons,
            pretty_print=config.pretty_print,
            sort_by=config.sort_by,
        )

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class PassRateActionRunner(ReportActionRunner):

    def _exit_code(self) -> int:
        if self.config.error_on_failures and self.base_report.get_summary_stats().failed > 0:
            return 1
        return 0

    def __call__(self, *args: Any, **kwds: Any) -> tuple[str, int]:
        if self.config.save_to_json:
            self.base_report.to_pass_rate_json_by_level(json_file=self.config.save_to_json,
                                                        level=self.config.pass_rate_level)
        return self.base_report.get_pass_rate_summary(), self._exit_code()


class TestsStatsActionRunner(ReportActionRunner):

    def __call__(self, *args: Any, **kwds: Any) -> str:
        return self.summary_detailed


class CompareReportsActionRunner(ReportActionRunner):

    def compare(self) -> pd.DataFrame:
        config = self.config
        omit_flags = {
            "omit_testsuite_name": config.omit_testsuite_name,
            "omit_test_module_name": config.omit_test_module_name,
            "omit_test_class_name": config.omit_test_class_name,
        }
        if config.report_grouping_level == TestGroupingLevel.TEST:
            effective_omit = omit_flags
        else:
            for flag_name, flag_value in omit_flags.items():
                if flag_value:
                    flag_cli = flag_name.replace("_", "-")
                    print(f"[WARNING] --{flag_cli} is only effective"
                          f" with --level test, ignoring", file=sys.stderr)
            effective_omit = {k: False for k in omit_flags}
        return TestReport.compare(
            self.reports,
            grouping_level=config.report_grouping_level,
            sort_by=SortByCompare(config.sort_by),
            compare_scope=config.compare_scope,
            **effective_omit,
        )

    def __call__(self, *args: Any, **kwds: Any) -> pd.DataFrame:
        return self.compare()


class ExportReportActionRunner(ReportActionRunner):

    def to_csv(self) -> None:
        self.base_report.to_csv(self.config.file_name)

    def __call__(self, *args: Any, **kwds: Any) -> None:
        return self.to_csv()


class DownloadReportsActionRunner(ActionRunner):

    def download_reports(self) -> None:
        config = self.config
        if config.latest_nightly_gh_run or config.nightly_run_id:
            GHANightlyTestReportProcessor(
                download_dir=config.download_dir,
                gh_run_id=config.nightly_run_id,
                repo=config.repo,
                branch=config.branch,
                artifact_pattern=config.artifact_pattern,
            ).download_test_reports()
        elif config.gh_run_id:
            GHABuildTestReportProcessor(
                download_dir=Path(config.download_dir),
                gh_run_id=config.gh_run_id,
                repo=config.repo,
                branch=config.branch,
                artifact_pattern=config.artifact_pattern,
            ).download_test_reports()
        else:
            raise ValueError(
                "Either nightly_run_id or gh_run_id should be provided or latest_nightly_gh_run should be set to True")

    def __call__(self, *args: Any, **kwds: Any) -> None:
        return self.download_reports()


class DownloadWheelsActionRunner(ActionRunner):  # pylint: disable=R0903

    def __call__(self, *args: Any, **kwds: Any) -> None:
        config = self.config
        python_version = config.python_version
        if not python_version and not config.download_for_all_pythons:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        downloader = GHAWheelDownloader(
            download_dir=config.download_dir,
            repo=config.repo,
            branch=config.branch,
            wheel_set=config.wheel_set,
            artifact_pattern=config.artifact_pattern,
            python_version=python_version if not config.download_for_all_pythons else None,
            gh_run_id=config.gh_run_id or None,
            latest_wf_run=config.latest_wf_run,
            latest_wf_run_pattern=config.latest_wf_run_pattern,
        )
        downloaded = downloader.download()
        for whl_path in downloaded:
            print(whl_path)
        if not downloaded:
            print("[WARNING] No wheels matched the specified filters", file=sys.stderr)


def run(config: Config) -> Any:  # pylint: disable=R0912
    if config.action == "download_wheels":
        return DownloadWheelsActionRunner(config=config)()
    if config.action == "download_reports":
        return DownloadReportsActionRunner(config=config)()
    if config.action == "export_to":
        if config.export_format == "csv":
            return ExportReportActionRunner(config=config)()
        raise NotImplementedError(f"Export format {config.export_format} is not implemented yet")
    if config.action == "pass_rate":
        summary, ex_code = PassRateActionRunner(config=config)()
        print(summary)
        sys.exit(ex_code)
    # Configure pandas display options for DataFrame output actions
    # Use option_context to avoid mutating global pandas state
    option_args: list[Any] = ["display.max_rows", None]
    if config.long_names:
        option_args.extend(["display.max_colwidth", None, "display.width", None])
    if config.action == "compare_reports":
        comparison = CompareReportsActionRunner(config=config)()
        with pd.option_context(*option_args):
            print(comparison)
        return comparison
    if config.action == "tests_stats":
        tests_stats = TestsStatsActionRunner(config=config)()
        with pd.option_context(*option_args):
            print(tests_stats)
        return tests_stats
    raise ValueError(f"Unknown action: {config.action}")


def main():
    run(Config.from_args())


if __name__ == "__main__":
    main()
