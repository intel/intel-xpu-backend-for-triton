"""Tests for convert_results.py -- CSV ingestion and history entry construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_monitor.convert_results import (
    build_history_entry,
    detect_platform,
    load_report_csvs,
    read_history,
    write_history,
)


class TestDetectPlatform:

    def test_pvc_max_1550(self):
        assert detect_platform("Intel Data Center GPU Max 1550") == "pvc"

    def test_pvc_max_1100(self):
        assert detect_platform("Intel Data Center GPU Max 1100") == "pvc"

    def test_bmg_b580(self):
        assert detect_platform("Intel Arc B580") == "bmg"

    def test_bmg_keyword(self):
        assert detect_platform("Some BMG Device") == "bmg"

    def test_unknown(self):
        assert detect_platform("NVIDIA A100") is None

    def test_empty_string(self):
        assert detect_platform("") is None


class TestLoadReportCsvs:

    def test_empty_dir(self, tmp_path: Path):
        df = load_report_csvs(tmp_path)
        assert df.empty

    def test_single_csv(self, tmp_path: Path):
        csv_file = tmp_path / "pvc-report.csv"
        csv_file.write_text("benchmark,compiler,params,tflops,gpu_device,datetime,agama_version,libigc1_version\n"
                            'gemm,triton,"{M:1024}",10.5,Intel Max 1550,2025-01-01,1.0,2.0\n')
        df = load_report_csvs(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["benchmark"] == "gemm"

    def test_multiple_csvs(self, tmp_path: Path):
        for name in ("a-report.csv", "b-report.csv"):
            (tmp_path / name).write_text("benchmark,compiler,params,tflops,gpu_device\n"
                                         "gemm,triton,{},10.0,PVC\n")
        df = load_report_csvs(tmp_path)
        assert len(df) == 2


class TestBuildHistoryEntry:

    def test_basic(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{M:1024}",
            "tflops": 10.5,
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entry = build_history_entry(df, run_id="123", commit_sha="abc", tag="ci")
        assert entry["run_id"] == "123"
        assert "gemm/triton/{M:1024}" in entry["results"]
        assert entry["results"]["gemm/triton/{M:1024}"]["tflops"] == 10.5

    def test_nan_tflops_filtered(self):
        """NaN tflops should be excluded from results (H1 fix)."""
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": float("nan"),
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entry = build_history_entry(df, run_id="1", commit_sha="a", tag="ci")
        assert "gemm/triton/{}" not in entry["results"]

    def test_numpy_nan_filtered(self):
        """numpy.float64 NaN should also be filtered (H1 fix)."""
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": np.float64("nan"),
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entry = build_history_entry(df, run_id="1", commit_sha="a", tag="ci")
        assert "gemm/triton/{}" not in entry["results"]

    def test_hbm_gbs_not_collected(self):
        """hbm_gbs should NOT appear in results (H3 fix)."""
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": 10.0,
            "hbm_gbs": 500.0,
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entry = build_history_entry(df, run_id="1", commit_sha="a", tag="ci")
        result = entry["results"]["gemm/triton/{}"]
        assert "tflops" in result
        assert "hbm_gbs" not in result

    def test_none_tflops_excluded(self):
        """None tflops should be excluded."""
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": None,
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entry = build_history_entry(df, run_id="1", commit_sha="a", tag="ci")
        assert "gemm/triton/{}" not in entry["results"]


class TestReadWriteHistory:

    def test_read_missing_file(self, tmp_path: Path):
        result = read_history(tmp_path / "nonexistent.json")
        assert result == []

    def test_read_non_list_json(self, tmp_path: Path):
        """A valid JSON file that is not a list should return empty list."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"key": "value"}')
        result = read_history(bad_file)
        assert result == []

    def test_write_creates_dirs(self, tmp_path: Path):
        path = tmp_path / "sub" / "dir" / "history.json"
        write_history(path, [{"test": True}])
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == [{"test": True}]

    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "history.json"
        original = [{"run_id": "1", "results": {}}]
        write_history(path, original)
        loaded = read_history(path)
        assert loaded == original
