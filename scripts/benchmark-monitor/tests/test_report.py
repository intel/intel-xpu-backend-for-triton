"""Tests for report_results.py -- GitHub issue/PR comment posting.

Covers formatting helpers, table generation, issue/PR comment building,
and the CI/PR handler integration paths with mocked gh CLI calls.
"""
# pylint: disable=wrong-import-position,redefined-outer-name

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the benchmark-monitor package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report_results import (  # noqa: E402
    BENCHMARK_MONITOR_MARKER, _build_issue_body, _build_pr_comment, _create_or_update_issue, _driver_change_notice,
    _find_existing_issue, _format_params, _improvement_table, _regression_table, _sort_improvements, _sort_regressions,
    handle_ci, handle_pr,
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


def _fake_gh_ok(args, **_kwargs):
    """Default mock: all gh calls succeed with empty output."""
    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


@pytest.fixture
def mock_gh(monkeypatch):
    """Mock _run_gh to capture calls without hitting GitHub API."""
    calls = []

    def fake_run_gh(args, **_kwargs):
        calls.append({"args": args, "input": _kwargs.get("input_text")})
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("report_results._run_gh", fake_run_gh)
    return calls


def _mock_gh_with_issues(monkeypatch, issues_json):
    """Mock _run_gh to return specific issue list JSON."""
    calls = []

    def fake(args, **_kwargs):
        calls.append({"args": args, "input": _kwargs.get("input_text")})
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(issues_json), stderr="")

    monkeypatch.setattr("report_results._run_gh", fake)
    return calls


# ---------------------------------------------------------------------------
# _format_params
# ---------------------------------------------------------------------------


class TestFormatParams:

    def test_valid_json(self):
        assert _format_params('{"M": 1024, "N": 512}') == "M=1024, N=512"

    def test_single_key(self):
        assert _format_params('{"K": 256}') == "K=256"

    def test_invalid_json_passthrough(self):
        assert _format_params("not-json") == "not-json"

    def test_empty_string(self):
        assert _format_params("") == ""

    def test_none_passthrough(self):
        assert _format_params(None) is None


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSortRegressions:

    def test_sort_by_drop_pct_ascending(self):
        items = [
            {"drop_pct": -5.0},
            {"drop_pct": -20.0},
            {"drop_pct": -10.0},
        ]
        result = _sort_regressions(items)
        assert [r["drop_pct"] for r in result] == [-20.0, -10.0, -5.0]

    def test_empty_list(self):
        assert _sort_regressions([]) == []


class TestSortImprovements:

    def test_sort_by_gain_pct_descending(self):
        items = [
            {"gain_pct": 5.0},
            {"gain_pct": 20.0},
            {"gain_pct": 10.0},
        ]
        result = _sort_improvements(items)
        assert [r["gain_pct"] for r in result] == [20.0, 10.0, 5.0]

    def test_empty_list(self):
        assert _sort_improvements([]) == []


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


class TestRegressionTable:

    def test_produces_valid_markdown(self):
        items = [_make_regression()]
        table = _regression_table(items)
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
        table = _regression_table(items, current_label="PR")
        assert "PR (TFlops)" in table

    def test_limit_truncation(self):
        items = [_make_regression(benchmark=f"bench_{i}") for i in range(5)]
        table = _regression_table(items, limit=2)
        assert "bench_0" in table
        assert "bench_1" in table
        assert "bench_4" not in table
        assert "... and 3 more" in table

    def test_no_truncation_when_under_limit(self):
        items = [_make_regression()]
        table = _regression_table(items, limit=10)
        assert "... and" not in table


class TestImprovementTable:

    def test_produces_valid_markdown(self):
        items = [_make_improvement()]
        table = _improvement_table(items)
        lines = table.strip().split("\n")
        assert len(lines) == 3
        assert "softmax" in lines[2]
        assert "N=512" in lines[2]

    def test_limit_truncation(self):
        items = [_make_improvement(benchmark=f"bench_{i}") for i in range(5)]
        table = _improvement_table(items, limit=3)
        assert "... and 2 more" in table


# ---------------------------------------------------------------------------
# _driver_change_notice
# ---------------------------------------------------------------------------


class TestDriverChangeNotice:

    def test_none_returns_empty(self):
        assert _driver_change_notice(None) == ""

    def test_empty_list_returns_empty(self):
        assert _driver_change_notice([]) == ""

    def test_single_change_with_arrow(self):
        changes = [{"field": "agama_version", "from": "1.0", "to": "2.0"}]
        result = _driver_change_notice(changes)
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
        result = _driver_change_notice(changes)
        assert "agama_version" in result
        assert "libigc1_version" in result


# ---------------------------------------------------------------------------
# _build_pr_comment
# ---------------------------------------------------------------------------


class TestBuildPrComment:

    def test_no_changes_shows_checkmark(self):
        report = {
            "gpus": {
                "pvc": {"total_checked": 50, "skipped": 2, "regressions": [], "improvements": []},
            },
        }
        body = _build_pr_comment(report)
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
        body = _build_pr_comment(report)
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
        body = _build_pr_comment(report)
        assert ":green_circle:" in body
        assert "Improvements (1 found)" in body

    def test_multiple_gpus(self):
        report = {
            "gpus": {
                "pvc": {"total_checked": 50, "skipped": 0, "regressions": [], "improvements": []},
                "bmg": {"total_checked": 30, "skipped": 0, "regressions": [], "improvements": []},
            },
        }
        body = _build_pr_comment(report)
        assert "PVC" in body
        assert "BMG" in body


# ---------------------------------------------------------------------------
# _build_issue_body
# ---------------------------------------------------------------------------


class TestBuildIssueBody:

    def test_complete_body(self):
        gpu_data = {
            "regressions": [_make_regression()],
            "improvements": [_make_improvement()],
            "total_checked": 50,
            "skipped": 3,
        }
        body = _build_issue_body("pvc", gpu_data, "12345", "https://example.com/run", "abc123", "2025-01-01")
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
        body = _build_issue_body("pvc", gpu_data, "1", "url", "sha", "date")
        assert "driver version change" in body

    def test_no_improvements_omits_section(self):
        gpu_data = {
            "regressions": [_make_regression()],
            "improvements": [],
            "total_checked": 10,
            "skipped": 0,
        }
        body = _build_issue_body("pvc", gpu_data, "1", "url", "sha", "date")
        assert "Improvements" not in body


# ---------------------------------------------------------------------------
# _find_existing_issue
# ---------------------------------------------------------------------------


class TestFindExistingIssue:

    def test_match_found(self, monkeypatch):
        issues = [
            {"number": 42, "title": "[Perf Regression] 3 benchmarks regressed on PVC"},
        ]
        _mock_gh_with_issues(monkeypatch, issues)
        result = _find_existing_issue("owner/repo", "pvc")
        assert result == 42

    def test_no_match(self, monkeypatch):
        issues = [
            {"number": 99, "title": "[Perf Regression] 2 benchmarks regressed on BMG"},
        ]
        _mock_gh_with_issues(monkeypatch, issues)
        result = _find_existing_issue("owner/repo", "pvc")
        assert result is None

    def test_empty_issues(self, monkeypatch):
        _mock_gh_with_issues(monkeypatch, [])
        result = _find_existing_issue("owner/repo", "pvc")
        assert result is None

    def test_gh_failure(self, monkeypatch):

        def fake(args, **_kwargs):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="error")

        monkeypatch.setattr("report_results._run_gh", fake)
        result = _find_existing_issue("owner/repo", "pvc")
        assert result is None


# ---------------------------------------------------------------------------
# _create_or_update_issue
# ---------------------------------------------------------------------------


class TestCreateOrUpdateIssue:

    def test_creates_new_issue(self, monkeypatch):
        # No existing issue found
        call_log = []

        def fake(args, **_kwargs):
            call_log.append(args)
            if "issue" in args and "list" in args:
                return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")
            if "issue" in args and "create" in args:
                return subprocess.CompletedProcess(args, 0, stdout="https://github.com/o/r/issues/1\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr("report_results._run_gh", fake)

        gpu_data = {"regressions": [_make_regression()], "improvements": [], "total_checked": 10, "skipped": 0}
        url = _create_or_update_issue("o/r", "pvc", gpu_data, "1", "url", "sha", "date")
        assert url == "https://github.com/o/r/issues/1"
        # Verify "issue create" was called
        create_calls = [c for c in call_log if "create" in c]
        assert len(create_calls) == 1

    def test_comments_on_existing_issue(self, monkeypatch):
        call_log = []
        issues_json = [{"number": 42, "title": "[Perf Regression] 1 benchmarks regressed on PVC"}]

        def fake(args, **_kwargs):
            call_log.append(args)
            if "issue" in args and "list" in args:
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(issues_json), stderr="")
            if "issue" in args and "comment" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr("report_results._run_gh", fake)

        gpu_data = {"regressions": [_make_regression()], "improvements": [], "total_checked": 10, "skipped": 0}
        url = _create_or_update_issue("o/r", "pvc", gpu_data, "1", "url", "sha", "date")
        assert url == "https://github.com/o/r/issues/42"
        comment_calls = [c for c in call_log if "comment" in c]
        assert len(comment_calls) == 1

    def test_no_regressions_returns_none(self, mock_gh):
        gpu_data = {"regressions": [], "improvements": [], "total_checked": 10, "skipped": 0}
        result = _create_or_update_issue("o/r", "pvc", gpu_data, "1", "url", "sha", "date")
        assert result is None
        assert len(mock_gh) == 0  # No gh calls made

    def test_driver_change_title(self, monkeypatch):
        call_log = []

        def fake(args, **_kwargs):
            call_log.append(args)
            if "issue" in args and "list" in args:
                return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")
            if "issue" in args and "create" in args:
                return subprocess.CompletedProcess(args, 0, stdout="https://github.com/o/r/issues/2\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr("report_results._run_gh", fake)

        gpu_data = {
            "regressions": [_make_regression()],
            "improvements": [],
            "total_checked": 10,
            "skipped": 0,
            "driver_change": [{"field": "agama_version", "from": "1.0", "to": "2.0"}],
        }
        _create_or_update_issue("o/r", "pvc", gpu_data, "1", "url", "sha", "date")
        create_calls = [c for c in call_log if "create" in c]
        assert len(create_calls) == 1
        # Title should include "Driver Change"
        title_arg_idx = create_calls[0].index("--title") + 1
        assert "Driver Change" in create_calls[0][title_arg_idx]


# ---------------------------------------------------------------------------
# handle_ci
# ---------------------------------------------------------------------------


class TestHandleCi:

    def test_creates_issue_for_regressions(self, monkeypatch):
        call_log = []

        def fake(args, **_kwargs):
            call_log.append(args)
            if "issue" in args and "list" in args:
                return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")
            if "issue" in args and "create" in args:
                return subprocess.CompletedProcess(args, 0, stdout="https://github.com/o/r/issues/1\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr("report_results._run_gh", fake)

        report = {
            "run_id": "123",
            "commit_sha": "abc",
            "datetime": "2025-01-01",
            "gpus": {
                "pvc": {
                    "regressions": [_make_regression()],
                    "improvements": [],
                    "total_checked": 10,
                    "skipped": 0,
                },
            },
        }
        handle_ci(report, run_url="https://example.com", repo="o/r")
        create_calls = [c for c in call_log if "create" in c]
        assert len(create_calls) == 1

    def test_skips_when_no_regressions(self, mock_gh):
        report = {
            "run_id": "123",
            "commit_sha": "abc",
            "datetime": "2025-01-01",
            "gpus": {
                "pvc": {
                    "regressions": [],
                    "improvements": [_make_improvement()],
                    "total_checked": 10,
                    "skipped": 0,
                },
            },
        }
        handle_ci(report, run_url="https://example.com", repo="o/r")
        assert len(mock_gh) == 0  # No gh calls


# ---------------------------------------------------------------------------
# handle_pr
# ---------------------------------------------------------------------------


class TestHandlePr:

    def test_creates_new_comment(self, monkeypatch):
        call_log = []

        def fake(args, **_kwargs):
            call_log.append(args)
            if "api" in args and "comments" in str(args):
                # No existing comment found
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr("report_results._run_gh", fake)

        report = {
            "gpus": {
                "pvc": {"total_checked": 10, "skipped": 0, "regressions": [], "improvements": []},
            },
        }
        handle_pr(report, pr_number="42", repo="o/r")
        # Should have a "pr comment" call
        pr_comment_calls = [c for c in call_log if "pr" in c and "comment" in c]
        assert len(pr_comment_calls) == 1
        assert "42" in pr_comment_calls[0]

    def test_updates_existing_comment(self, monkeypatch):
        call_log = []

        def fake(args, **_kwargs):
            call_log.append(args)
            if "api" in args and "comments" in str(args) and "PATCH" not in args:
                # Return existing comment ID
                return subprocess.CompletedProcess(args, 0, stdout="99887766", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr("report_results._run_gh", fake)

        report = {
            "gpus": {
                "pvc": {"total_checked": 10, "skipped": 0, "regressions": [], "improvements": []},
            },
        }
        handle_pr(report, pr_number="42", repo="o/r")
        # Should have a PATCH call to update the comment
        patch_calls = [c for c in call_log if "PATCH" in c]
        assert len(patch_calls) == 1
        assert "99887766" in str(patch_calls[0])
