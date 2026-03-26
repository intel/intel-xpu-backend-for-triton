"""Tests for benchmark_monitor.formatting — markdown tables, issue bodies, PR comments.

Port of test_report.py adapted to the new module structure.
GitHub API interaction tests (TestFindExistingIssue, TestCreateOrUpdateIssue,
TestHandleCi, TestHandlePr) are intentionally NOT ported since the reporting
module was removed.
"""
# pylint: disable=too-few-public-methods

from __future__ import annotations

from benchmark_monitor.formatting import (
    BENCHMARK_MONITOR_MARKER,
    build_issue_body,
    build_pr_comment,
    driver_change_notice,
    format_issue_title,
    format_params,
    format_summary,
    improvement_table,
    regression_table,
    sort_improvements,
    sort_regressions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_regression(**overrides):
    """Build a regression dict with sensible defaults."""
    base = {
        "benchmark": "matmul",
        "params": '{"M": 1024}',
        "drop_pct": -10.0,
        "baseline_median": 100.0,
        "current_tflops": 90.0,
        "modified_z": -3.0,
    }
    base.update(overrides)
    return base


def _make_improvement(**overrides):
    """Build an improvement dict with sensible defaults."""
    base = {
        "benchmark": "softmax",
        "params": '{"N": 512}',
        "gain_pct": 15.0,
        "baseline_median": 80.0,
        "current_tflops": 92.0,
        "modified_z": 3.5,
    }
    base.update(overrides)
    return base


# ===================================================================
# format_params
# ===================================================================


class TestFormatParams:

    def test_valid_json(self):
        assert format_params('{"M": 1024, "N": 512}') == "M=1024, N=512"

    def test_single_key(self):
        assert format_params('{"K": 256}') == "K=256"

    def test_invalid_json_passthrough(self):
        assert format_params("not-json") == "not-json"

    def test_empty_string(self):
        assert format_params("") == ""

    def test_none_passthrough(self):
        assert format_params(None) is None


# ===================================================================
# Sorting
# ===================================================================


class TestSortRegressions:

    def test_sort_by_drop_pct_ascending(self):
        items = [
            {"drop_pct": -5.0},
            {"drop_pct": -20.0},
            {"drop_pct": -10.0},
        ]
        result = sort_regressions(items)
        assert [r["drop_pct"] for r in result] == [-20.0, -10.0, -5.0]

    def test_empty_list(self):
        assert not sort_regressions([])


class TestSortImprovements:

    def test_sort_by_gain_pct_descending(self):
        items = [
            {"gain_pct": 5.0},
            {"gain_pct": 20.0},
            {"gain_pct": 10.0},
        ]
        result = sort_improvements(items)
        assert [r["gain_pct"] for r in result] == [20.0, 10.0, 5.0]

    def test_empty_list(self):
        assert not sort_improvements([])


# ===================================================================
# Table generation
# ===================================================================


class TestRegressionTable:

    def test_produces_valid_markdown(self):
        items = [_make_regression()]
        table = regression_table(items)
        lines = table.strip().split("\n")
        # Header + separator + 1 data row
        assert len(lines) == 3
        assert "Benchmark" in lines[0]
        assert "Baseline (TFlops)" in lines[0]
        assert "Change" in lines[0]
        assert "matmul" in lines[2]
        assert "M=1024" in lines[2]

    def test_custom_current_label(self):
        items = [_make_regression()]
        table = regression_table(items, current_label="PR")
        assert "PR (TFlops)" in table

    def test_limit_truncation(self):
        items = [_make_regression(benchmark=f"bench_{i}") for i in range(5)]
        table = regression_table(items, limit=2)
        assert "bench_0" in table
        assert "bench_1" in table
        assert "bench_4" not in table
        assert "... and 3 more" in table

    def test_no_truncation_when_under_limit(self):
        items = [_make_regression()]
        table = regression_table(items, limit=10)
        assert "... and" not in table


class TestImprovementTable:

    def test_produces_valid_markdown(self):
        items = [_make_improvement()]
        table = improvement_table(items)
        lines = table.strip().split("\n")
        assert len(lines) == 3
        assert "softmax" in lines[2]
        assert "N=512" in lines[2]

    def test_limit_truncation(self):
        items = [_make_improvement(benchmark=f"bench_{i}") for i in range(5)]
        table = improvement_table(items, limit=3)
        assert "... and 2 more" in table


# ===================================================================
# driver_change_notice
# ===================================================================


class TestDriverChangeNotice:

    def test_none_returns_empty(self):
        assert driver_change_notice(None) == ""

    def test_empty_list_returns_empty(self):
        assert driver_change_notice([]) == ""

    def test_single_change_with_arrow(self):
        changes = [{"field": "agama_version", "from": "1.0", "to": "2.0"}]
        result = driver_change_notice(changes)
        assert "driver version change" in result
        assert "\u2192" in result  # arrow
        assert "agama_version" in result
        assert "1.0" in result
        assert "2.0" in result

    def test_multiple_changes(self):
        changes = [
            {"field": "agama_version", "from": "1.0", "to": "2.0"},
            {"field": "libigc1_version", "from": "3.0", "to": "4.0"},
        ]
        result = driver_change_notice(changes)
        assert "agama_version" in result
        assert "libigc1_version" in result


# ===================================================================
# format_issue_title
# ===================================================================


class TestFormatIssueTitle:

    def test_basic_title(self):
        title = format_issue_title("pvc", 3)
        assert title == "[Perf Regression] 3 benchmarks regressed on PVC"

    def test_with_driver_change(self):
        changes = [{"field": "agama_version", "from": "1.0", "to": "2.0"}]
        title = format_issue_title("pvc", 1, driver_change=changes)
        assert "Driver Change" in title
        assert "PVC" in title

    def test_bmg_gpu(self):
        title = format_issue_title("bmg", 5)
        assert "BMG" in title

    def test_unknown_gpu_uppercased(self):
        title = format_issue_title("xyz", 2)
        assert "XYZ" in title


# ===================================================================
# build_pr_comment
# ===================================================================


class TestBuildPrComment:

    def test_no_changes_shows_checkmark(self):
        report = {
            "gpus": {
                "pvc": {"total_checked": 50, "skipped": 2, "regressions": [], "improvements": []},
            },
        }
        body = build_pr_comment(report)
        assert BENCHMARK_MONITOR_MARKER in body
        assert ":white_check_mark:" in body
        assert "PVC" in body

    def test_with_regressions_has_details(self):
        report = {
            "gpus": {
                "pvc": {
                    "total_checked": 50,
                    "skipped": 0,
                    "regressions": [_make_regression()],
                    "improvements": [],
                },
            },
        }
        body = build_pr_comment(report)
        assert "<details>" in body
        assert ":red_circle:" in body
        assert "Regressions (1 found)" in body

    def test_with_improvements_has_details(self):
        report = {
            "gpus": {
                "bmg": {
                    "total_checked": 30,
                    "skipped": 1,
                    "regressions": [],
                    "improvements": [_make_improvement()],
                },
            },
        }
        body = build_pr_comment(report)
        assert ":green_circle:" in body
        assert "Improvements (1 found)" in body

    def test_multiple_gpus(self):
        report = {
            "gpus": {
                "pvc": {"total_checked": 50, "skipped": 0, "regressions": [], "improvements": []},
                "bmg": {"total_checked": 30, "skipped": 0, "regressions": [], "improvements": []},
            },
        }
        body = build_pr_comment(report)
        assert "PVC" in body
        assert "BMG" in body


# ===================================================================
# build_issue_body
# ===================================================================


class TestBuildIssueBody:

    def test_complete_body(self):
        gpu_data = {
            "regressions": [_make_regression()],
            "improvements": [_make_improvement()],
            "total_checked": 50,
            "skipped": 3,
        }
        body = build_issue_body("pvc", gpu_data, "12345", "https://example.com/run", "abc123", "2025-01-01")
        assert "Performance Regression Detected" in body
        assert "#12345" in body
        assert "abc123" in body
        assert "2025-01-01" in body
        assert "PVC" in body
        assert "Regressions (1 found" in body
        assert "Improvements (1 found)" in body
        assert "matmul" in body

    def test_with_driver_change(self):
        gpu_data = {
            "regressions": [_make_regression()],
            "improvements": [],
            "total_checked": 10,
            "skipped": 0,
            "driver_change": [{"field": "agama_version", "from": "1.0", "to": "2.0"}],
        }
        body = build_issue_body("pvc", gpu_data, "1", "url", "sha", "date")
        assert "driver version change" in body

    def test_no_improvements_omits_section(self):
        gpu_data = {
            "regressions": [_make_regression()],
            "improvements": [],
            "total_checked": 10,
            "skipped": 0,
        }
        body = build_issue_body("pvc", gpu_data, "1", "url", "sha", "date")
        assert "Improvements" not in body


# ===================================================================
# format_summary
# ===================================================================


class TestFormatSummary:

    def test_basic_summary(self):
        report = {
            "run_id": "123",
            "commit_sha": "abc",
            "gpus": {
                "pvc": {
                    "total_checked": 10,
                    "skipped": 2,
                    "regressions": [_make_regression()],
                    "improvements": [],
                },
            },
        }
        text = format_summary(report, tag="ci")
        assert "Benchmark Report" in text
        assert "tag=ci" in text
        assert "run=123" in text
        assert "PVC" in text
        assert "Regressions: 1" in text

    def test_no_changes(self):
        report = {
            "run_id": "1",
            "commit_sha": "x",
            "gpus": {
                "bmg": {
                    "total_checked": 5,
                    "skipped": 0,
                    "regressions": [],
                    "improvements": [],
                },
            },
        }
        text = format_summary(report, tag="ci")
        assert "No significant changes" in text

    def test_with_improvements(self):
        report = {
            "run_id": "1",
            "commit_sha": "x",
            "gpus": {
                "pvc": {
                    "total_checked": 5,
                    "skipped": 0,
                    "regressions": [],
                    "improvements": [_make_improvement()],
                },
            },
        }
        text = format_summary(report, tag="ci")
        assert "Improvements: 1" in text

    def test_empty_gpus(self):
        report = {"run_id": "1", "commit_sha": "x", "gpus": {}}
        text = format_summary(report, tag="ci")
        assert "Benchmark Report" in text
