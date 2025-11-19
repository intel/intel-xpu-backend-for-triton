from __future__ import annotations

from typing import Any
from enum import Enum
from collections import defaultdict
from dataclasses import field, fields, dataclass

import os
import pathlib
import glob

import re

import json
import csv

import xml.etree.ElementTree as stdET

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

    def to_named_dict(self, raw_data_only: bool = False) -> dict[str, dict[str, int | float]]:
        if raw_data_only:
            return {
                self.name: {
                    "passed": self.passed,
                    "failed": self.failed,
                    "skipped": self.skipped,
                    "xfailed": self.xfailed,
                }
            }
        return {self.name: self.to_metrics_dict()}

    def to_json(self, pretty_print: bool = False, named_dict: bool = True) -> str:
        res_dict = self.to_named_dict() if named_dict else self.to_metrics_dict()
        if pretty_print:
            return json.dumps(res_dict, indent=1)
        return json.dumps(res_dict, separators=(",", ":"))


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


class TestGroupingLevel(Enum):
    REPORT = "report"
    TESTSUITE = "testsuite"
    TEST = "test"


@dataclass
class TestCase:  #  pylint: disable=too-many-instance-attributes
    # intel
    testsuite: str
    # test.unit.intel.test_block_load
    classname: str
    # test_block_load_dpas_layout[True-int8-256-64]
    name: str
    # test_block_load
    module: str = field(init=False)
    # [True-int8-256-64]
    variant: str = field(init=False)
    # test_block_load_dpas_layout
    test: str = field(init=False)
    # test/unit/intel/test_block_load/test_block_load.py::test_block_load_dpas_layout[True-int8-256-64]
    path: str = field(init=False)
    # test/unit/intel/test_block_load/test_block_load.py::test_block_load_dpas_layout
    path_without_variant: str = field(init=False)
    # intel::test/unit/intel/test_block_load/test_block_load.py::test_block_load_dpas_layout[True-int8-256-64]
    key: str = field(init=False)

    def __post_init__(self):
        raw_name = self.name
        test_classname = self.classname
        test_subpaths = test_classname.rsplit(".", 1)
        if len(test_subpaths) == 1:
            self.path = ""
            self.module = test_subpaths[0]
        else:
            self.path = test_subpaths[0]
            self.module = test_subpaths[1]
        index = raw_name.find("[")
        if index != -1:
            self.test = raw_name[:index]
            self.variant = raw_name[index:]
        else:
            self.test = raw_name
            self.variant = ""

        self.path_without_variant = f"{self.classname.replace('.', '/')}/{self.module}.py::{self.test}"
        self.path = f"{self.path_without_variant}{self.variant}"
        self.key = f"{self.testsuite}::{self.path}"


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
        pattern = re.compile(r"(?:.*/)?(?:([^/]+)/)?([^/]*?)\.py::([-\w\[\]]+)")
        match = pattern.match(self.testname)
        if match:
            _, module, test = match.groups()
        else:
            raise ValueError(f"Cannot extract short name from testname: {self.testname}")
        return f"{self.testsuite}::{module}.{test}"

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
        stats = ReportStats(name=f"{self.testsuite}::{self.testname}")
        for test_case in self.test_cases:
            if test_case.result == RunResult.PASSED:
                stats.passed += 1
            if test_case.result == RunResult.FAILED:
                stats.failed += 1
            if test_case.result == RunResult.SKIPPED:
                stats.skipped += 1
            if test_case.result == RunResult.XFAILED:
                stats.xfailed += 1
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
                f"Test contains test cases from multiple tests:{self.testname} - {test_case.path_without_variant}")
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
            print("\n".join("WARNING: No junit xml files - " + str(folder) for folder in empty_subfolders))

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
            if (existing_result == RunResult.PASSED and new_result == RunResult.FAILED
                    or existing_result == RunResult.FAILED and new_result == RunResult.PASSED):
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
                    if existing_test_case.result == RunResult.SKIPPED and test_case.result in [
                            RunResult.PASSED, RunResult.FAILED
                    ]:
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

    def get_summary(
        self,
        list_test_instances: bool = False,
        list_failure_reasons: bool = False,
        grouping_level: TestGroupingLevel = TestGroupingLevel.TEST,
    ) -> str:
        match grouping_level:
            case TestGroupingLevel.REPORT:
                return self.get_pass_rate_summary()
            case TestGroupingLevel.TESTSUITE:
                testsuites: dict[str, dict[str, Test]] = defaultdict(dict)
                for test_key, test in self.tests.items():
                    testsuites[test.testsuite][test_key] = test
                testsuite_stats = [
                    TestReport(tests=tests, name=testsuite).get_summary_stats().to_json()
                    for testsuite, tests in testsuites.items()
                ]
                return "\n".join(testsuite_stats)
            case TestGroupingLevel.TEST:
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
        tests_dict: list[dict[str, Any]] = [{
            "testsuite": test_case.testsuite,
            "classname": test_case.classname,
            "module": test_case.module,
            "test": test_case.test,
            "variant": test_case.variant,
            "time": test_case.time,
            "status": test_case.result.value,
        } for test_case in tests]
        with open(
                csv_file,
                mode="w",
                newline="",
                encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(tests_dict[0].keys()))
            writer.writeheader()
            writer.writerows(tests_dict)

    @classmethod
    def compare(  # pylint: disable=R0914
        cls,
        reports: list[TestReport],
        grouping_level: TestGroupingLevel = TestGroupingLevel.TESTSUITE,
    ) -> pd.DataFrame:
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
                            tests=tests, name=testsuite).get_summary_stats().to_named_dict(raw_data_only=True)
                    reports_stats.append(report_stats)
                case TestGroupingLevel.TEST:
                    for test_key, test in report.tests.items():
                        report_stats = report_stats | TestReport(tests={
                            test.short_name: test
                        }, name=test.short_name).get_summary_stats().to_named_dict(raw_data_only=True)
                    reports_stats.append(report_stats)
                case _:
                    raise ValueError(f"Unsupported grouping level {grouping_level}")

        columns = list(next(iter(reports_stats[0].values())).keys())

        left_r = pd.DataFrame.from_dict(reports_stats[0], orient="index")
        right_r = pd.DataFrame.from_dict(reports_stats[1], orient="index")

        left_r, right_r = left_r.align(right_r, join="outer")

        diff_abs = right_r - left_r

        comparison = pd.concat(
            {
                "r1": left_r,
                "r2": right_r,
                "Δ": diff_abs,
            },
            axis=1,
        ).swaplevel(axis=1).sort_index(axis=1, level=0).round(0)

        comparison = comparison.reindex(columns=columns, level=0)

        summary = comparison.apply(pd.to_numeric, errors="coerce").sum(numeric_only=True)
        summary.name = "Σ"
        comparison_with_total = pd.concat([comparison, summary.to_frame().T])

        def to_int_or_na(val):
            if pd.isna(val):
                return "NA"
            return int(round(val))

        return comparison_with_total.map(to_int_or_na)

    @classmethod
    def from_reports_folder(
        cls,
        folder: str,
        tests_with_multiple_testsuites: bool,
        ignore_testsuites_filter: list[str],
        pattern_matcher: PatternMatcher | None = None,
    ) -> TestReport:

        def _generate_tutorials_report():
            pass

        return TestReport(
            cls._extract_test_reports(
                folder,
                tests_with_multiple_testsuites,
                ignore_testsuites_filter,
                pattern_matcher=pattern_matcher,
            ))
