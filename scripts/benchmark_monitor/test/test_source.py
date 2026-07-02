"""Tests for benchmark_monitor.source — report data source implementations."""
# pylint: disable=too-few-public-methods

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from unittest.mock import patch

from benchmark_monitor.source import (
    GHArtifactSource,
    LocalCSVSource,
    _read_report_csvs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_report_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV file from a list of row dicts."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _base_row(**overrides) -> dict:
    """Create a base CSV row with sensible defaults."""
    row = {
        "benchmark": "matmul",
        "compiler": "triton",
        "params": '{"M": 1024}',
        "tflops": "100.5",
        "gpu_device": "Intel Data Center GPU Max 1550",
        "agama_version": "1.0.0",
        "libigc1_version": "2.0.0",
        "datetime": "2025-01-01T00:00:00Z",
    }
    row.update(overrides)
    return row


# ===================================================================
# _read_report_csvs (internal helper)
# ===================================================================


class TestReadReportCsvs:

    def test_empty_dir(self, tmp_path: Path):
        df = _read_report_csvs(tmp_path)
        assert df.empty

    def test_single_csv(self, tmp_path: Path):
        _write_report_csv(tmp_path / "pvc-report.csv", [_base_row()])
        df = _read_report_csvs(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["benchmark"] == "matmul"

    def test_multiple_csvs(self, tmp_path: Path):
        _write_report_csv(tmp_path / "a-report.csv", [_base_row(benchmark="gemm")])
        _write_report_csv(tmp_path / "b-report.csv", [_base_row(benchmark="softmax")])
        df = _read_report_csvs(tmp_path)
        assert len(df) == 2

    def test_non_report_csv_ignored(self, tmp_path: Path):
        _write_report_csv(tmp_path / "other.csv", [_base_row()])
        df = _read_report_csvs(tmp_path)
        assert df.empty


# ===================================================================
# LocalCSVSource
# ===================================================================


class TestLocalCSVSource:

    def test_fetch_with_csvs(self, tmp_path: Path):
        _write_report_csv(tmp_path / "pvc-report.csv", [_base_row()])
        source = LocalCSVSource(tmp_path)
        df = source.fetch()
        assert len(df) == 1
        assert df.iloc[0]["benchmark"] == "matmul"

    def test_fetch_empty_directory(self, tmp_path: Path):
        source = LocalCSVSource(tmp_path)
        df = source.fetch()
        assert df.empty

    def test_fetch_multiple_files(self, tmp_path: Path):
        _write_report_csv(tmp_path / "a-report.csv", [_base_row(benchmark="gemm")])
        _write_report_csv(tmp_path / "b-report.csv", [_base_row(benchmark="softmax")])
        source = LocalCSVSource(tmp_path)
        df = source.fetch()
        assert len(df) == 2

    def test_fetch_with_multiple_rows(self, tmp_path: Path):
        rows = [
            _base_row(benchmark="gemm", tflops="10.0"),
            _base_row(benchmark="softmax", tflops="5.0"),
        ]
        _write_report_csv(tmp_path / "bench-report.csv", rows)
        source = LocalCSVSource(tmp_path)
        df = source.fetch()
        assert len(df) == 2


# ===================================================================
# GHArtifactSource (mocked)
# ===================================================================


class TestGHArtifactSource:

    def test_fetch_success(self, tmp_path: Path):  # pylint: disable=unused-argument
        """Test successful artifact download with mocked _run_gh."""

        def fake_run_gh(args):
            # Simulate successful download by creating CSV files in the dest dir
            # The dest dir is extracted from the args
            dir_idx = args.index("--dir") + 1
            dest = Path(args[dir_idx])
            dest.mkdir(parents=True, exist_ok=True)
            _write_report_csv(dest / "bench-report.csv", [_base_row()])
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("benchmark_monitor.source.GHArtifactSource._run_gh", side_effect=fake_run_gh):
            source = GHArtifactSource(run_id=12345, artifact_name="reports", repo="org/repo")
            df = source.fetch()
            assert len(df) == 1
            assert df.iloc[0]["benchmark"] == "matmul"

    def test_fetch_failure(self):
        """Test that download failure returns empty DataFrame."""

        def fake_run_gh(args):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="error")

        with patch("benchmark_monitor.source.GHArtifactSource._run_gh", side_effect=fake_run_gh):
            source = GHArtifactSource(run_id=99999, artifact_name="reports", repo="org/repo")
            df = source.fetch()
            assert df.empty

    def test_fetch_no_csvs(self):
        """Test that empty artifact directory returns empty DataFrame."""

        def fake_run_gh(args):
            dir_idx = args.index("--dir") + 1
            dest = Path(args[dir_idx])
            dest.mkdir(parents=True, exist_ok=True)
            # No CSV files created
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("benchmark_monitor.source.GHArtifactSource._run_gh", side_effect=fake_run_gh):
            source = GHArtifactSource(run_id=12345)
            df = source.fetch()
            assert df.empty

    def test_constructor_defaults(self):
        source = GHArtifactSource(run_id=42)
        assert source.run_id == 42
        assert source.artifact_name == "benchmark-reports"
        assert source.repo == "intel/intel-xpu-backend-for-triton"
