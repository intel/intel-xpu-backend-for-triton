from __future__ import annotations

import csv
import datetime
import glob
import json
import os
import pathlib
import platform
import re
import sys
import xml.etree.ElementTree as stdET
from collections import defaultdict
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar

import defusedxml.ElementTree as ET
import pandas as pd

from .pattern_matcher import PatternMatcher

XMLElement = stdET.Element


@dataclass
class ReportStats:  # pylint: disable=R0801
    """Report stats."""
    name: str = ""
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    fixme: int = 0
    time: float = 0.0

    RESULT_FIELDS: ClassVar[list[str]] = ["passed", "failed", "skipped", "xfailed"]
    METRIC_FIELDS: ClassVar[list[str]] = ["time", "pass_rate_without_xfailed"]
    COMPARE_FIELDS: ClassVar[list[str]] = ["passed", "failed", "skipped", "xfailed", "time"]

    @property
    def total(self):
        return self.passed + self.failed + self.skipped + self.xfailed

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

    def __add__(self, other: ReportStats) -> ReportStats:
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            v1 = getattr(self, f.name)
            v2 = getattr(other, f.name)
            if isinstance(v1, (int, float)):
                kwargs[f.name] = v1 + v2
            elif isinstance(v1, str):
                kwargs[f.name] = v1
            else:
                raise TypeError(f"Unsupported field type for {f.name}: {type(v1)}")
        return ReportStats(**kwargs)

    def __radd__(self, other: ReportStats) -> ReportStats:
        return self.__add__(other)

    def to_metrics_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for f in fields(self):
            field_value = getattr(self, f.name)
            if not isinstance(field_value, (int, float)):
                continue
            result[f.name] = field_value
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, property):
                result[name] = getattr(self, name)
        return result

    def to_named_dict(self, fields_filter: list[str] | None = None) -> dict[str, dict[str, int | float]]:
        if fields_filter is None:
            return {self.name: self.to_metrics_dict()}
        res_fields = {}
        for res_field in self.to_metrics_dict():
            if res_field in fields_filter:
                res_fields[res_field] = getattr(self, res_field)
        return {self.name: res_fields}

    def to_json(self, pretty_print: bool = False, named_dict: bool = True) -> str:
        res_dict = self.to_named_dict() if named_dict else self.to_metrics_dict()
        if pretty_print:
            return json.dumps(res_dict, indent=1)
        return json.dumps(res_dict, separators=(",", ":"))


# pylint: disable=R0801


class RunResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    XFAILED = "xfailed"


class FailureReason(Enum):
    # https://github.com/testmoapp/junitxml
    FAILURE = "failure"
    ERROR = "error"
    SKIPPED = "pytest.skip"
    XFAILED = "pytest.xfail"


# pylint: enable=R0801


class TestGroupingLevel(Enum):
    REPORT = "report"
    TESTSUITE = "testsuite"
    TEST = "test"


class CompareScope(Enum):
    ANY = "any"
    R1_ONLY = "r1-only"
    R2_ONLY = "r2-only"
    BOTH = "both"


class SortByStats(str, Enum):
    NAME = "name"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    TIME = "time"
    PASS_RATE = "pass_rate_without_xfailed"


class SortByCompare(str, Enum):
    NAME = "name"
    PASSED_R1 = "passed.r1"
    PASSED_R2 = "passed.r2"
    PASSED_DELTA = "passed.Δ"
    FAILED_R1 = "failed.r1"
    FAILED_R2 = "failed.r2"
    FAILED_DELTA = "failed.Δ"
    SKIPPED_R1 = "skipped.r1"
    SKIPPED_R2 = "skipped.r2"
    SKIPPED_DELTA = "skipped.Δ"
    XFAILED_R1 = "xfailed.r1"
    XFAILED_R2 = "xfailed.r2"
    XFAILED_DELTA = "xfailed.Δ"
    TIME_R1 = "time.r1"
    TIME_R2 = "time.r2"
    TIME_DELTA = "time.Δ"
    TIME_PCT_DELTA = "time.%Δ"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            if value.endswith(".%delta"):
                canonical = value.replace(".%delta", ".%Δ")
                for member in cls:
                    if member.value == canonical:
                        return member
            if value.endswith(".delta"):
                canonical = value.replace(".delta", ".Δ")
                for member in cls:
                    if member.value == canonical:
                        return member
        return None


@dataclass
class TestCase:  #  pylint: disable=too-many-instance-attributes
    # intel
    testsuite: str
    # test.unit.intel.test_block_io
    classname: str
    # test_block_io[256-64-float16-row_major-True-True-False]
    name: str
    # test_block_io
    module: str = field(init=False)
    # [256-64-float16-row_major-True-True-False]
    variant: str = field(init=False)
    # test_block_io
    test: str = field(init=False)
    # test/unit/intel/test_block_io/test_block_io.py::test_block_io[256-64-float16-row_major-True-True-False]
    path: str = field(init=False)
    # test/unit/intel/test_block_io/test_block_io.py::test_block_io
    path_without_variant: str = field(init=False)
    # intel::test/unit/intel/test_block_io/test_block_io.py::test_block_io[256-64-float16-row_major-True-True-False]
    key: str = field(init=False)
    # TestGraph_c55uxh25dhggnpnm666db2a5xn6yhinukzlhsmg37d7bjkwvwxv5XPU (empty if no class)
    test_class: str = field(init=False)
    # pytest-friendly name: test/unit/intel/test_block_io.py::test_block_io[256-64-float16-row_major-True-True-False]
    pytest_name: str = field(init=False)

    def __post_init__(self):
        raw_name = self.name
        test_classname = self.classname
        test_subpaths = test_classname.rsplit(".", 1)
        self.test_class = ""
        if len(test_subpaths) == 1:
            self.module = test_subpaths[0]
        else:
            last_part = test_subpaths[1]
            preceding_part = test_subpaths[0].rsplit(".", 1)[-1]
            # Detect class name: the last segment is a class (not a module) when
            # it doesn't follow the test_*.py naming convention and the preceding
            # segment does (confirming it's the actual module).
            if not last_part.startswith("test_") and preceding_part.startswith("test_"):
                self.test_class = last_part
                self.module = preceding_part
            else:
                self.module = last_part
        index = raw_name.find("[")
        if index != -1:
            self.test = raw_name[:index]
            self.variant = raw_name[index:]
        else:
            self.test = raw_name
            self.variant = ""

        if self.test_class:
            module_path = test_classname[:test_classname.rfind(".")].replace(".", "/")
            self.path_without_variant = f"{module_path}/{self.module}.py::{self.test_class}::{self.test}"
        else:
            self.path_without_variant = f"{self.classname.replace('.', '/')}/{self.module}.py::{self.test}"
        self.path = f"{self.path_without_variant}{self.variant}"
        self.key = f"{self.testsuite}::{self.path}"

        # pytest-friendly name: path/to/module.py::[ClassName::]test[variant]
        if self.test_class:
            module_dotted = test_classname[:test_classname.rfind(".")]
        else:
            module_dotted = test_classname
        pytest_file = module_dotted.replace(".", "/") + ".py"
        if self.test_class:
            self.pytest_name = f"{pytest_file}::{self.test_class}::{self.test}{self.variant}"
        else:
            self.pytest_name = f"{pytest_file}::{self.test}{self.variant}"


@dataclass
class TestCaseRunResult:
    result: RunResult = RunResult.PASSED
    time: float = 0
    result_reason: str = ""
    error_message: str = ""
    error_text: str = ""


@dataclass
class TestCaseWithResult(
        TestCaseRunResult,
        TestCase,
):

    @classmethod
    def from_xml(cls, testsuite: str, xml_element: XMLElement) -> TestCaseWithResult:
        raw_name = xml_element.get("name", "")
        classname = xml_element.get("classname", "")

        test_time = float(xml_element.get("time", "0"))
        result_reason = ""
        error_message = ""
        error_text: str | None = ""

        child_tags = [child_tag for child_tag in list(xml_element) if child_tag.tag in ["failure", "skipped", "error"]]
        run_result = RunResult.PASSED
        if len(child_tags) > 1:
            raise ValueError("Unexpected number of child tags for the test")
        if len(child_tags) == 1:
            child_element = child_tags[0]
            child_tag = child_element.tag
            error_text = child_element.text if hasattr(child_element, "text") else ""
            error_message = child_element.attrib.get("message", "")
            if child_tag in ["skipped"]:
                result_reason = child_element.get("type", "pytest.skip")
                match result_reason:
                    case "pytest.skip":
                        run_result = RunResult.SKIPPED
                    case "pytest.xfail":
                        run_result = RunResult.XFAILED
                    case _:
                        raise ValueError(f"Unsupported skip type: {result_reason}")
            else:
                run_result = RunResult.FAILED
                result_reason = child_tag

        return TestCaseWithResult(
            testsuite=testsuite,
            classname=classname,
            name=raw_name,
            result=run_result,
            time=test_time,
            result_reason=result_reason,
            error_message=error_message,
            error_text=str(error_text),
        )


@dataclass
class Test:
    test_cases: list[TestCaseWithResult] = field(default_factory=list)
    testsuite: str = ""
    testname: str = ""

    @property
    def short_name(self) -> str:
        pattern = re.compile(r"(?:.*/)?(?:([^/]+)/)?([^/]*?)\.py::([-\w\[\]]+(?:::[-\w\[\]]+)?)")
        match = pattern.match(self.testname)
        if match:
            _, module, test = match.groups()
        else:
            raise ValueError(f"Cannot extract short name from testname: {self.testname}")
        return f"{self.testsuite}::{module}.{test}"

    @property
    def pytest_name(self) -> str:
        """Pytest-friendly test node id (without variant)."""
        if self.test_cases:
            tc = self.test_cases[0]
            # Strip variant from pytest_name to get the base test id
            if tc.variant and tc.pytest_name.endswith(tc.variant):
                return tc.pytest_name[:-len(tc.variant)]
            return tc.pytest_name
        return self.testname

    def get_reason_messages(self) -> str:
        reasons_by_result: dict[RunResult, set[str]] = {}
        for test_case in self.test_cases:
            if test_case.result != RunResult.PASSED:
                reasons_by_result[test_case.result] = reasons_by_result.get(test_case.result, set()).union({test_case.error_message})  # yapf: disable
        if len(reasons_by_result) == 0:
            return ""
        reason_messages = ""
        for result, reasons in reasons_by_result.items():
            reason_message_text = "\n".join(str(item) for item in reasons)
            reason_messages += f"{result} reasons({len(reasons)}):\n{reason_message_text}\n"
        return reason_messages

    def get_failed_test_variants(self) -> str:
        test_variants: list[str] = []
        for test_case in self.test_cases:
            if test_case.result == RunResult.FAILED:
                test_variants.append(f"{test_case.path}{test_case.variant}")
        if len(test_variants) == 0:
            return ""
        failed_test_cases_text = "\n".join(str(item) for item in test_variants)
        return f"Failed test cases({len(test_variants)}):\n{failed_test_cases_text}"

    def get_test_variants(self) -> str:
        test_variants_by_result: dict[RunResult, list[str]] = defaultdict(list)
        for test_case in self.test_cases:
            # test_variants_by_result[test_case.result] = (
            #     test_variants_by_result.get(test_case.result, []).append(test_case.path)
            # )
            test_variants_by_result[test_case.result].append(test_case.path)
        if len(test_variants_by_result) == 0:
            return ""
        test_variants_str = ""
        for result, test_variants in test_variants_by_result.items():
            test_variants_text = "\n".join(str(item) for item in test_variants)
            test_variants_str += f"{result}({len(test_variants)}):\n{test_variants_text}\n"
        return test_variants_str

    def get_stats(self) -> ReportStats:
        stats = ReportStats(name=f"{self.testsuite}::{self.pytest_name}")
        for test_case in self.test_cases:
            if test_case.result == RunResult.PASSED:
                stats.passed += 1
            if test_case.result == RunResult.FAILED:
                stats.failed += 1
            if test_case.result == RunResult.SKIPPED:
                stats.skipped += 1
            if test_case.result == RunResult.XFAILED:
                stats.xfailed += 1
            stats.time += test_case.time
        return stats

    def add_test_case(self, test_case: TestCaseWithResult):
        if test_case.testsuite == "":
            raise ValueError(f"Test case have no test suite:{test_case.key}")
        if self.testsuite and test_case.testsuite != self.testsuite:
            raise ValueError(
                f"Test contains test cases from multiple testsuites:{self.testsuite} - {test_case.testsuite}."
                "Use either '--tests-with-multiple-testsuites' flag or ignore one of the testsuites by setting '--ignore-testsuite <testsuite_name>' filter."
            )
        self.testsuite = test_case.testsuite
        if self.testname and test_case.path_without_variant != self.testname:
            raise ValueError(
                f"Test contains test cases from multiple tests:{self.testname} - {test_case.path_without_variant}"
            )
        if test_case.classname == "" and test_case.name == "":
            print(f"[WARNING] Skipping test case with no classname and name in {self.testsuite}", file=sys.stderr)
            return
        self.testname = test_case.path_without_variant
        self.test_cases.append(test_case)

    def filter_variants(
        self,
        status_filter: list[str],
        ignore_testsuites: list[str],
        testsuite: str | None = None,
        testname_wo_variant: str | None = None,
    ) -> Test | None:
        filtered_test_cases: list[TestCaseWithResult] = []
        # Add check whether test is in a test suite plus test
        for test_case in self.test_cases:
            # yapf: disable
            if (   # pylint: disable=too-many-boolean-expressions
                test_case.result.value in status_filter and self.testsuite not in ignore_testsuites
                and (testsuite is None or self.testsuite == testsuite)
                and (testname_wo_variant is None or self.testname == testname_wo_variant)
            ):
                filtered_test_cases.append(test_case)
                # print(self.testname)
            # yapf: enable
        if len(filtered_test_cases) == 0:
            return None
        return Test(
            test_cases=filtered_test_cases,
            testsuite=self.testsuite,
            testname=self.testname,
        )


@dataclass
class TestReport:
    tests: dict[str, Test]
    name: str = ""
    testsuite_filter: str = ""
    testname_filter: str = ""

    @classmethod
    def _extract_test_reports(  # pylint: disable=too-many-locals
        cls,
        search_folder: str,
        tests_with_multiple_testsuites: bool,
        ignore_testsuites_filter: list[str],
        pattern_matcher: PatternMatcher | None = None,
    ) -> dict[str, Test]:
        search_folder_path = pathlib.Path(search_folder)
        if not search_folder_path.exists():
            raise ValueError(f"Reports folder {search_folder} does not exist")
        search_pattern = os.path.join(search_folder, "**", "*.xml")
        all_paths = glob.glob(search_pattern, recursive=True)

        if len(all_paths) == 0:
            raise ValueError(f"No junit xml found in the reports folder {search_folder}")

        empty_subfolders = [
            folder for folder in search_folder_path.iterdir() if folder.is_dir() and not any(folder.glob("*.xml"))
        ]
        if len(empty_subfolders) > 0:
            print(
                "\n".join("WARNING: No junit xml files - " + str(folder) for folder in empty_subfolders),
                file=sys.stderr
            )

        tests: dict[str, Test] = {}
        for file_path in all_paths:
            relative_path = pathlib.Path(file_path).relative_to(search_folder_path)
            if len(relative_path.parts) > 1 and pattern_matcher and not pattern_matcher.matches(str(relative_path)):
                print(f"Skipping file {file_path} due to pattern matcher filtering")
                continue
            test_suite = os.path.splitext(os.path.basename(file_path))[0]
            if test_suite in ignore_testsuites_filter:
                print(f"Skipping testsuite {test_suite} due to ignore testsuite filter")
                continue
            tree = ET.parse(file_path)
            root = tree.getroot()
            for test_case_xml in root.iter("testcase"):
                test_case = TestCaseWithResult.from_xml(
                    testsuite=test_suite,
                    xml_element=test_case_xml,
                )
                if tests_with_multiple_testsuites:
                    key = f"{test_suite}::{test_case.path_without_variant}"
                else:
                    key = test_case.path_without_variant
                test = tests.get(key, Test())
                test.add_test_case(test_case)
                if len(test.test_cases) == 1:
                    tests[key] = test
        return tests

    def filter_test_instances(
        self,
        status_filter: list[str],
        testsuite: str | None,
        testname_wo_variant: str | None,
        ignore_testsuites: list[str],
    ) -> TestReport:
        filtered_tests: dict[str, Test] = {}

        for test_key, test in self.tests.items():
            if filtered_test := test.filter_variants(
                    status_filter=status_filter,
                    testsuite=testsuite,
                    testname_wo_variant=testname_wo_variant,
                    ignore_testsuites=ignore_testsuites,
            ):
                filtered_tests[test_key] = filtered_test
        return TestReport(filtered_tests, str(testsuite), str(testname_wo_variant))

    def merge_test_results(self) -> TestReport:

        def _check_flaky_results(case_name: str, existing_result: RunResult, new_result: RunResult) -> bool:
            if ((existing_result == RunResult.PASSED and new_result == RunResult.FAILED)
                    or (existing_result == RunResult.FAILED and new_result == RunResult.PASSED)):
                print(f"[WARNING] Flaky test detected: {case_name}")
                return True
            return False

        merged_tests: dict[str, Test] = {}

        for test_key, test in self.tests.items():
            best_test_cases: dict[str, TestCaseWithResult] = {}
            for test_case in test.test_cases:
                existing_test_case = best_test_cases.get(test_case.variant, None)
                if existing_test_case is None:
                    best_test_cases[test_case.variant] = test_case
                else:
                    # Determine the best result
                    _check_flaky_results(existing_test_case.key, existing_test_case.result, test_case.result)
                    if existing_test_case.result == RunResult.PASSED:
                        continue
                    if existing_test_case.result == RunResult.FAILED and test_case.result == RunResult.PASSED:
                        best_test_cases[test_case.variant] = test_case
                    if existing_test_case.result == RunResult.SKIPPED and test_case.result in [RunResult.PASSED,
                                                                                               RunResult.FAILED]:
                        best_test_cases[test_case.variant] = test_case
                    if existing_test_case.result == RunResult.XFAILED and test_case.result in [
                            RunResult.PASSED, RunResult.FAILED, RunResult.SKIPPED
                    ]:
                        best_test_cases[test_case.variant] = test_case
            merged_tests[test_key] = Test(
                test_cases=list(best_test_cases.values()),
                testsuite=test.testsuite,
                testname=test.testname,
            )
        return TestReport(merged_tests, self.testsuite_filter, self.testname_filter)

    def list_test_instances(
        self,
        print_data: bool = True,
    ) -> list[TestCaseWithResult]:
        test_instances: list[TestCaseWithResult] = []
        for key, test in self.tests.items():
            for test_case in test.test_cases:
                test_instances.append(test_case)
                if print_data:
                    print(f"{key}{test_case.variant}")
        return test_instances

    def check_for_multiple_test_results(self) -> bool:
        multiple_test_results_found = False
        for test in self.tests.values():
            seen_variants: set[str] = set()
            for test_case in test.test_cases:
                if test_case.variant in seen_variants:
                    multiple_test_results_found = True
                else:
                    seen_variants.add(test_case.variant)
        return multiple_test_results_found

    def get_summary_stats(self) -> ReportStats:
        agg_stats = ReportStats(name=self.name)
        for test in self.tests.values():
            agg_stats += test.get_stats()
            # test_stats = test.get_stats()
            # for stat_key, stat_value in test_stats.items():
            #     agg_stats[stat_key] = agg_stats[stat_key] + stat_value
        return agg_stats

    def get_pass_rate_summary(self) -> str:
        return self.get_summary_stats().to_json(pretty_print=True, named_dict=False)

    @classmethod
    def _df_w_total_row(cls, raw_df: pd.DataFrame) -> pd.DataFrame:
        summary = raw_df.apply(pd.to_numeric, errors="coerce").sum(numeric_only=True)
        summary.name = "Σ"

        return pd.concat([raw_df, summary.to_frame().T])

    @classmethod
    def _get_report_dfs(  # pylint: disable=dangerous-default-value, too-many-locals
        cls,
        reports: list[TestReport],
        grouping_level: TestGroupingLevel = TestGroupingLevel.TESTSUITE,
        fields_filter: list[str] = ReportStats.RESULT_FIELDS,
        sort_by: str = "name",
    ) -> tuple[list[pd.DataFrame], list[str]]:
        reports_stats: list[dict[str, dict[str, int | float]]] = []
        for report in reports:
            report_stats: dict[str, dict[str, int | float]] = {}
            match grouping_level:
                case TestGroupingLevel.REPORT:
                    raise ValueError(f"Unsupported grouping level {grouping_level}")
                case TestGroupingLevel.TESTSUITE:
                    testsuites: dict[str, dict[str, Test]] = defaultdict(dict)
                    for test_key, test in report.tests.items():
                        testsuites[test.testsuite][test_key] = test
                    for testsuite, tests in testsuites.items():
                        report_stats = report_stats | TestReport(
                            tests=tests, name=testsuite
                        ).get_summary_stats().to_named_dict(fields_filter=fields_filter)
                    reports_stats.append(report_stats)
                case TestGroupingLevel.TEST:
                    for test_key, test in report.tests.items():
                        pytest_key = f"{test.testsuite}::{test.pytest_name}"
                        report_stats = report_stats | TestReport(
                            tests={
                                pytest_key: test
                            }, name=pytest_key
                        ).get_summary_stats().to_named_dict(fields_filter=fields_filter)
                    reports_stats.append(report_stats)
                case _:
                    raise ValueError(f"Unsupported grouping level {grouping_level}")
        columns = list(next(iter(reports_stats[0].values())).keys())
        report_dfs = [pd.DataFrame.from_dict(report_stats, orient="index") for report_stats in reports_stats]
        for report_df in report_dfs:
            if sort_by == "name":
                report_df.sort_index(inplace=True)
            else:
                report_df.sort_values(by=sort_by, inplace=True, ascending=False)
        return report_dfs, columns

    def _get_summary_df(
        self,
        grouping_level: TestGroupingLevel,
        sort_by: str = "name",
    ) -> pd.DataFrame:
        try:
            SortByStats(sort_by)
        except ValueError as e:
            raise ValueError(f"Unsupported sort_by field: {sort_by}") from e
        summary_df = self._df_w_total_row(
            self._get_report_dfs(
                [self],
                grouping_level=grouping_level,
                fields_filter=ReportStats.RESULT_FIELDS + ReportStats.METRIC_FIELDS,
                sort_by=sort_by,
            )[0][0],
        )
        summary_df = summary_df.rename(columns={"pass_rate_without_xfailed": "pass_rate"})

        passed = summary_df.loc["Σ", "passed"]
        failed = summary_df.loc["Σ", "failed"]
        skipped = summary_df.loc["Σ", "skipped"]
        xfailed = summary_df.loc["Σ", "xfailed"]
        total = passed + failed + skipped + xfailed
        denom = total - xfailed
        summary_df.loc["Σ", "pass_rate"] = round(100 * passed / denom, 2)

        for columns in ReportStats.RESULT_FIELDS:
            summary_df[columns] = summary_df[columns].astype(int)
        summary_df["pass_rate"] = summary_df["pass_rate"].round(2).astype(str) + "%"
        return summary_df

    def get_summary(  # pylint: disable=too-many-positional-arguments, too-many-arguments
        self,
        list_test_instances: bool = False,
        list_failure_reasons: bool = False,
        grouping_level: TestGroupingLevel = TestGroupingLevel.TEST,
        pretty_print: bool = False,
        sort_by: str = "name",
    ) -> str | pd.DataFrame:
        match grouping_level:
            case TestGroupingLevel.REPORT:
                return self.get_pass_rate_summary()
            case TestGroupingLevel.TESTSUITE:
                if pretty_print:
                    return self._get_summary_df(
                        grouping_level=grouping_level,
                        sort_by=sort_by,
                    )
                testsuites: dict[str, dict[str, Test]] = defaultdict(dict)
                for test_key, test in self.tests.items():
                    testsuites[test.testsuite][test_key] = test
                test_reports = [TestReport(tests=tests, name=testsuite) for testsuite, tests in testsuites.items()]
                return "\n".join([test_report.get_summary_stats().to_json() for test_report in test_reports])
            case TestGroupingLevel.TEST:
                if pretty_print:
                    return self._get_summary_df(
                        grouping_level=grouping_level,
                        sort_by=sort_by,
                    )
                test_stats = ""
                for test_key, test in self.tests.items():
                    test_stats += test.get_stats().to_json() + "\n"
                    if list_test_instances:
                        test_stats += test.get_test_variants() + "\n"
                    if list_failure_reasons:
                        test_stats += test.get_reason_messages() + "\n"
                return test_stats
            case _:
                raise ValueError(f"Unsupported grouping level {grouping_level}")

    def to_csv(self, csv_file: str):
        tests = self.list_test_instances(False)
        tests_dict: list[dict[str, Any]] = [
            {
                "testsuite": test_case.testsuite,
                "classname": test_case.classname,
                "module": test_case.module,
                "test": test_case.test,
                "variant": test_case.variant,
                "time": test_case.time,
                "status": test_case.result.value,
            } for test_case in tests
        ]
        with open(
                csv_file,
                mode="w",
                newline="",
                encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(tests_dict[0].keys()))
            writer.writeheader()
            writer.writerows(tests_dict)

    def get_pass_rate_json_data(self, testsuite_name: str = "all") -> dict:
        """Generate pass rate JSON data dictionary."""
        stats = self.get_summary_stats()
        return {
            "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "os": platform.system(),
            "git_ref": os.getenv("GITHUB_REF_NAME", ""),
            "git_sha": os.getenv("GITHUB_SHA", ""),
            "libigc1_version": os.getenv("LIBIGC1_VERSION", ""),
            "level_zero_version": os.getenv("LEVEL_ZERO_VERSION", ""),
            "agama_version": os.getenv("AGAMA_VERSION", ""),
            "gpu_device": os.getenv("GPU_DEVICE", ""),
            "python_version": platform.python_version(),
            "pytorch_version": os.getenv("PYTORCH_VERSION", ""),
            "testsuite": testsuite_name,
            "passed": stats.passed,
            "failed": stats.failed,
            "skipped": stats.skipped,
            "xfailed": stats.xfailed,
            "total": stats.total,
            "fixme": stats.fixme,
            "pass_rate_1": stats.pass_rate,
            "pass_rate_2": stats.pass_rate_without_xfailed,
        }  # yapf: disable

    def to_pass_rate_json_by_level(self, json_file: str, level: str = "all"):  # pylint: disable=R0801
        """
        Save pass rate JSON report based on grouping level.

        Args:
            json_file: Path to output file
            level: Grouping level ("all" or "testsuite")
                - "all": Creates a single JSON file with aggregate stats (one iteration)
                - "testsuite": Creates a JSONL file with one JSON per testsuite (multiple iterations)
        """
        # Group tests by testsuite (or "all" for level="all")
        testsuites: dict[str, dict[str, Test]] = {}

        if level == "all":
            # For "all" level: group all tests under single "all" key
            testsuites["all"] = self.tests
        elif level == "testsuite":
            # For "testsuite" level: group by actual testsuite name
            for test_key, test in self.tests.items():
                if test.testsuite not in testsuites:
                    testsuites[test.testsuite] = {}
                testsuites[test.testsuite][test_key] = test
        else:
            raise ValueError(f"Unsupported level: {level}. Must be 'all' or 'testsuite'")

        # Write file - same logic for both cases, just different number of iterations
        with open(json_file, "w", encoding="utf-8") as f:
            for testsuite_name, testsuite_tests in testsuites.items():
                # Create a TestReport for this testsuite (or "all")
                testsuite_report = TestReport(tests=testsuite_tests, name=testsuite_name)

                # Get data dict using helper method
                data = testsuite_report.get_pass_rate_json_data(testsuite_name=testsuite_name)

                # Write using json.dumps() - formatted for "all", single-line for "testsuite"
                if level == "all":
                    f.write(json.dumps(data, indent=2))
                else:  # level == "testsuite"
                    f.write(json.dumps(data) + "\n")

    @staticmethod
    def _minify_name(
        name: str,
        omit_testsuite: bool = False,
        omit_module: bool = False,
        omit_class: bool = False,
    ) -> str:
        """Minify a test name by stripping testsuite, module path, and/or class name.

        Index format: <testsuite>::<path>/<module>.py[::<class>]::<test>
        The class segment is optional.
        """
        if name in ("Σ", ""):
            return name
        parts = name.split("::")
        result: list[str] = []
        seen_module = False
        for part in parts:
            is_module = part.endswith(".py") or (("/" in part or "." in part) and not part.startswith("test_"))
            if is_module:
                seen_module = True
                if not omit_module:
                    result.append(part)
            elif part.startswith("test_"):
                result.append(part)
            elif not seen_module:
                if not omit_testsuite:
                    result.append(part)
            elif not omit_class:
                result.append(part)
        return "::".join(result)

    @classmethod
    def compare(  # pylint: disable=R0912, R0914, R0915, too-many-arguments, too-many-positional-arguments
        cls,
        reports: list[TestReport],
        grouping_level: TestGroupingLevel = TestGroupingLevel.TESTSUITE,
        sort_by: SortByCompare = SortByCompare.NAME,
        compare_scope: CompareScope = CompareScope.ANY,
        omit_testsuite_name: bool = False,
        omit_test_module_name: bool = False,
        omit_test_class_name: bool = False,
    ) -> pd.DataFrame:
        reports_stats, columns = cls._get_report_dfs(reports, grouping_level, fields_filter=ReportStats.COMPARE_FIELDS)
        left_r = reports_stats[0]
        right_r = reports_stats[1]
        left_r, right_r = left_r.align(right_r, join="outer")
        diff_abs = right_r.fillna(0) - left_r.fillna(0)

        # Compute percentage delta for time: 100 * (r2 - r1) / r1
        time_pct = pd.DataFrame(index=left_r.index)
        if "time" in left_r.columns:
            left_time = left_r["time"].fillna(0)
            right_time = right_r["time"].fillna(0)
            time_pct["time"] = (100.0 * (right_time - left_time) / left_time.where(left_time != 0)).round(2)

        comparison = pd.concat(
            {
                "r1": left_r,
                "r2": right_r,
                "Δ": diff_abs,
                "%Δ": time_pct,
            },
            axis=1,
        ).swaplevel(axis=1).sort_index(
            axis=1, level=0
        )
        comparison = comparison.reindex(columns=columns, level=0)

        # Reorder sources within each metric: r1, r2, Δ, %Δ
        source_order = ["r1", "r2", "Δ", "%Δ"]
        ordered_cols = []
        for metric in columns:
            for source in source_order:
                if (metric, source) in comparison.columns:
                    ordered_cols.append((metric, source))
        comparison = comparison[ordered_cols]

        # Round count metrics to integers, keep time with 2-decimal precision
        count_metrics = ["passed", "failed", "skipped", "xfailed"]
        count_cols = [col for col in comparison.columns if col[0] in count_metrics]
        time_cols = [col for col in comparison.columns if col[0] == "time"]
        if count_cols:
            comparison[count_cols] = comparison[count_cols].round(0)
        if time_cols:
            comparison[time_cols] = comparison[time_cols].round(2)

        # Filter by compare scope
        if compare_scope != CompareScope.ANY:
            r1_has = left_r.notna().any(axis=1)
            r2_has = right_r.notna().any(axis=1)
            match compare_scope:
                case CompareScope.R1_ONLY:
                    mask = r1_has & ~r2_has
                case CompareScope.R2_ONLY:
                    mask = ~r1_has & r2_has
                case CompareScope.BOTH:
                    mask = r1_has & r2_has
                case _:
                    raise ValueError(f"Invalid compare_scope: '{compare_scope}'")
            comparison = comparison.loc[mask]

        # Sort
        if sort_by != SortByCompare.NAME:
            metric, source = sort_by.value.rsplit(".", 1)
            comparison = comparison.sort_values(by=(metric, source), ascending=False)

        # Add total row
        comparison_with_total = cls._df_w_total_row(comparison)

        # Recompute %Δ for total row from summed time values
        if ("time", "%Δ") in comparison_with_total.columns:
            total_r1 = comparison_with_total.loc["Σ", ("time", "r1")]
            total_r2 = comparison_with_total.loc["Σ", ("time", "r2")]
            if pd.notna(total_r1) and total_r1 != 0:
                comparison_with_total.loc["Σ", ("time", "%Δ")] = round(100.0 * (total_r2 - total_r1) / total_r1, 2)
            else:
                comparison_with_total.loc["Σ", ("time", "%Δ")] = float("nan")

        # Format: time as float, others as int, NaN as "NA"
        def _to_int_or_na(val):
            if pd.isna(val):
                return "NA"
            return int(round(val))

        def _to_float_or_na(val):
            if pd.isna(val):
                return "NA"
            return round(float(val), 2)

        def _to_pct_or_na(val):
            if pd.isna(val):
                return "NA"
            return f"{round(float(val), 2)}%"

        comparison_result = comparison_with_total.copy()
        for col in comparison_result.columns:
            metric, source = col[0], col[1]
            if source == "%Δ":
                comparison_result[col] = comparison_result[col].map(_to_pct_or_na)
            elif metric == "time":
                comparison_result[col] = comparison_result[col].map(_to_float_or_na)
            else:
                comparison_result[col] = comparison_result[col].map(_to_int_or_na)

        # Insert group headers only when sorting by name (default)
        def _insert_group_headers(df: pd.DataFrame) -> pd.DataFrame:
            mask = df.index != "Σ"
            df_no_total = df[mask]
            df_total = df[~mask]
            idx = df_no_total.index.to_series().str.split("::", n=1, expand=True)
            groups = idx[0]
            tests = idx[1]
            frames = []
            for group, subdf in df_no_total.groupby(groups):
                header = pd.DataFrame([[""] * df.shape[1]], columns=df.columns, index=[group])
                subdf2 = subdf.copy()
                subdf2.index = tests.loc[subdf.index]
                frames.append(header)
                frames.append(subdf2)
            if not df_total.empty:
                frames.append(df_total)
            return pd.concat(frames)

        # Minify index names (before group headers, which depend on :: splitting)
        has_omit = omit_testsuite_name or omit_test_module_name or omit_test_class_name
        if has_omit:
            comparison_result.index = [
                cls._minify_name(name, omit_testsuite_name, omit_test_module_name, omit_test_class_name)
                for name in comparison_result.index
            ]

        # Insert group headers only when sorting by name and no omit flags
        use_group_headers = (
            grouping_level == TestGroupingLevel.TEST and sort_by == SortByCompare.NAME and not has_omit
        )
        if use_group_headers:
            comparison_result = _insert_group_headers(comparison_result)

        return comparison_result

    @classmethod
    def from_reports_folder(
        cls,
        folder: str,
        tests_with_multiple_testsuites: bool,
        ignore_testsuites_filter: list[str],
        pattern_matcher: PatternMatcher | None = None,
    ) -> TestReport:
        return TestReport(
            cls._extract_test_reports(
                folder,
                tests_with_multiple_testsuites,
                ignore_testsuites_filter,
                pattern_matcher=pattern_matcher,
            )
        )
