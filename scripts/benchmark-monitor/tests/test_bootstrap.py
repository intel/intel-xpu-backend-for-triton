"""Tests for bootstrap_history.py -- historical benchmark data bootstrapping.

Covers GPU detection from CSV files and report parsing into history entries.
"""
# pylint: disable=wrong-import-position

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure the benchmark-monitor package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bootstrap_history import detect_gpu, parse_reports

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV file with headers from the first row's keys."""
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


# ---------------------------------------------------------------------------
# detect_gpu
# ---------------------------------------------------------------------------


class TestDetectGpu:

    def test_pvc_max_1550(self, tmp_path):
        _write_csv(tmp_path / "bench-report.csv", [_base_row(gpu_device="Intel Data Center GPU Max 1550")])
        assert detect_gpu(tmp_path) == "pvc"

    def test_pvc_max_1100(self, tmp_path):
        _write_csv(tmp_path / "bench-report.csv", [_base_row(gpu_device="Intel Data Center GPU Max 1100")])
        assert detect_gpu(tmp_path) == "pvc"

    def test_bmg_b580(self, tmp_path):
        _write_csv(tmp_path / "bench-report.csv", [_base_row(gpu_device="Intel Arc B580")])
        assert detect_gpu(tmp_path) == "bmg"

    def test_bmg_keyword(self, tmp_path):
        _write_csv(tmp_path / "bench-report.csv", [_base_row(gpu_device="BMG Test Device")])
        assert detect_gpu(tmp_path) == "bmg"

    def test_unknown_device(self, tmp_path):
        _write_csv(tmp_path / "bench-report.csv", [_base_row(gpu_device="Unknown GPU Model")])
        assert detect_gpu(tmp_path) is None

    def test_empty_directory(self, tmp_path):
        assert detect_gpu(tmp_path) is None

    def test_non_report_csv_ignored(self, tmp_path):
        # File doesn't match *-report.csv glob
        _write_csv(tmp_path / "other.csv", [_base_row(gpu_device="Intel Data Center GPU Max 1550")])
        assert detect_gpu(tmp_path) is None


# ---------------------------------------------------------------------------
# parse_reports
# ---------------------------------------------------------------------------


class TestParseReports:

    def test_normal_csv(self, tmp_path):
        rows = [
            _base_row(benchmark="matmul", params='{"M": 1024}', tflops="100.5"),
            _base_row(benchmark="softmax", params='{"N": 512}', tflops="50.2"),
        ]
        _write_csv(tmp_path / "bench-report.csv", rows)

        run_info = {"headSha": "abc123", "createdAt": "2025-01-01T00:00:00Z"}
        entry = parse_reports(tmp_path, run_id=42, run_info=run_info)

        assert entry is not None
        assert entry["run_id"] == "42"
        assert entry["commit_sha"] == "abc123"
        assert entry["tag"] == "ci"
        assert entry["agama_version"] == "1.0.0"
        assert entry["libigc1_version"] == "2.0.0"
        assert len(entry["results"]) == 2
        assert "matmul/triton/{\"M\": 1024}" in entry["results"]
        assert entry["results"]["matmul/triton/{\"M\": 1024}"]["tflops"] == 100.5

    def test_no_triton_rows(self, tmp_path):
        rows = [_base_row(compiler="torch", tflops="80.0")]
        _write_csv(tmp_path / "bench-report.csv", rows)

        entry = parse_reports(tmp_path, run_id=1, run_info={})
        assert entry is None

    def test_empty_directory(self, tmp_path):
        entry = parse_reports(tmp_path, run_id=1, run_info={})
        assert entry is None

    def test_empty_tflops_skipped(self, tmp_path):
        rows = [_base_row(tflops="")]
        _write_csv(tmp_path / "bench-report.csv", rows)

        entry = parse_reports(tmp_path, run_id=1, run_info={})
        assert entry is None

    def test_invalid_tflops_skipped(self, tmp_path):
        rows = [_base_row(tflops="not-a-number")]
        _write_csv(tmp_path / "bench-report.csv", rows)

        entry = parse_reports(tmp_path, run_id=1, run_info={})
        assert entry is None

    def test_hbm_gbs_not_collected(self, tmp_path):
        """After H3 fix, hbm_gbs should NOT be collected in results."""
        rows = [_base_row()]
        _write_csv(tmp_path / "bench-report.csv", rows)

        entry = parse_reports(tmp_path, run_id=1, run_info={})
        assert entry is not None
        for _key, metrics in entry["results"].items():
            assert "hbm_gbs" not in metrics, "hbm_gbs should not be collected"

    def test_metadata_from_first_row(self, tmp_path):
        rows = [
            _base_row(agama_version="1.0", libigc1_version="2.0"),
            _base_row(agama_version="9.9", libigc1_version="9.9"),
        ]
        _write_csv(tmp_path / "bench-report.csv", rows)

        entry = parse_reports(tmp_path, run_id=1, run_info={})
        assert entry is not None
        # Metadata should come from the first row
        assert entry["agama_version"] == "1.0"
        assert entry["libigc1_version"] == "2.0"

    def test_multiple_csv_files(self, tmp_path):
        _write_csv(tmp_path / "a-report.csv", [_base_row(benchmark="matmul", tflops="100.0")])
        _write_csv(tmp_path / "b-report.csv", [_base_row(benchmark="softmax", tflops="50.0")])

        entry = parse_reports(tmp_path, run_id=1, run_info={})
        assert entry is not None
        assert len(entry["results"]) == 2

    def test_datetime_fallback_to_run_info(self, tmp_path):
        """When the CSV datetime column is missing, run_info createdAt is used."""
        # Create a CSV without the datetime column to trigger the fallback
        row = _base_row()
        del row["datetime"]
        _write_csv(tmp_path / "bench-report.csv", [row])

        run_info = {"createdAt": "2025-06-15T12:00:00Z"}
        entry = parse_reports(tmp_path, run_id=1, run_info=run_info)
        assert entry is not None
        assert entry["datetime"] == "2025-06-15T12:00:00Z"
