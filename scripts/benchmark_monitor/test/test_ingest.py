"""Tests for benchmark_monitor.ingest — CSV-to-BenchmarkEntry parsing."""
# pylint: disable=too-few-public-methods

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark_monitor.ingest import parse_reports

# ===================================================================
# parse_reports — valid data
# ===================================================================


class TestParseReportsValid:

    def test_basic(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{M:1024}",
            "tflops": 10.5,
            "gpu_device": "Intel Data Center GPU Max 1550",
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entries = parse_reports(df, run_id="123", commit_sha="abc", tag="ci")
        assert "pvc" in entries
        entry = entries["pvc"]
        assert entry.run_id == "123"
        assert "gemm/triton/{M:1024}" in entry.results
        assert entry.results["gemm/triton/{M:1024}"]["tflops"] == 10.5

    def test_multiple_platforms(self):
        df = pd.DataFrame([
            {
                "benchmark": "gemm",
                "compiler": "triton",
                "params": "{M:1024}",
                "tflops": 10.5,
                "gpu_device": "Intel Data Center GPU Max 1550",
                "datetime": "2025-01-01",
                "agama_version": "1.0",
                "libigc1_version": "2.0",
            },
            {
                "benchmark": "softmax",
                "compiler": "triton",
                "params": "{N:512}",
                "tflops": 5.0,
                "gpu_device": "Intel Arc B580",
                "datetime": "2025-01-01",
                "agama_version": "1.0",
                "libigc1_version": "2.0",
            },
        ])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        assert "pvc" in entries
        assert "bmg" in entries

    def test_multiple_benchmarks_same_platform(self):
        df = pd.DataFrame([
            {
                "benchmark": "gemm",
                "compiler": "triton",
                "params": "{M:1024}",
                "tflops": 10.0,
                "gpu_device": "Intel Data Center GPU Max 1550",
                "datetime": "2025-01-01",
                "agama_version": "1.0",
                "libigc1_version": "2.0",
            },
            {
                "benchmark": "softmax",
                "compiler": "triton",
                "params": "{N:512}",
                "tflops": 5.0,
                "gpu_device": "Intel Data Center GPU Max 1550",
                "datetime": "2025-01-01",
                "agama_version": "1.0",
                "libigc1_version": "2.0",
            },
        ])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        assert len(entries["pvc"].results) == 2

    def test_metadata_from_first_row(self):
        df = pd.DataFrame([
            {
                "benchmark": "gemm",
                "compiler": "triton",
                "params": "{}",
                "tflops": 10.0,
                "gpu_device": "Intel Data Center GPU Max 1550",
                "datetime": "2025-01-01",
                "agama_version": "1.0",
                "libigc1_version": "2.0",
            },
            {
                "benchmark": "softmax",
                "compiler": "triton",
                "params": "{}",
                "tflops": 5.0,
                "gpu_device": "Intel Data Center GPU Max 1550",
                "datetime": "2025-01-02",
                "agama_version": "9.9",
                "libigc1_version": "9.9",
            },
        ])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        entry = entries["pvc"]
        assert entry.agama_version == "1.0"
        assert entry.libigc1_version == "2.0"


# ===================================================================
# parse_reports — empty / invalid data
# ===================================================================


class TestParseReportsEmpty:

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        assert not entries

    def test_missing_compiler_column(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "params": "{}",
            "tflops": 10.0,
            "gpu_device": "Intel Data Center GPU Max 1550",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        assert not entries

    def test_no_triton_rows(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "torch",
            "params": "{}",
            "tflops": 80.0,
            "gpu_device": "Intel Data Center GPU Max 1550",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        assert not entries


# ===================================================================
# parse_reports — NaN / None filtering
# ===================================================================


class TestParseReportsFiltering:

    def test_nan_tflops_filtered(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": float("nan"),
            "gpu_device": "Intel Data Center GPU Max 1550",
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        # NaN tflops means no metrics, so no results -> possibly empty entry or no entry
        if "pvc" in entries:
            assert "gemm/triton/{}" not in entries["pvc"].results

    def test_numpy_nan_filtered(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": np.float64("nan"),
            "gpu_device": "Intel Data Center GPU Max 1550",
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        if "pvc" in entries:
            assert "gemm/triton/{}" not in entries["pvc"].results

    def test_none_tflops_excluded(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": None,
            "gpu_device": "Intel Data Center GPU Max 1550",
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        if "pvc" in entries:
            assert "gemm/triton/{}" not in entries["pvc"].results

    def test_hbm_gbs_not_collected(self):
        """hbm_gbs should NOT appear in results."""
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": 10.0,
            "hbm_gbs": 500.0,
            "gpu_device": "Intel Data Center GPU Max 1550",
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        result = entries["pvc"].results["gemm/triton/{}"]
        assert "tflops" in result
        assert "hbm_gbs" not in result

    def test_unknown_gpu_device_skipped(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": 10.0,
            "gpu_device": "NVIDIA A100",
            "datetime": "2025-01-01",
            "agama_version": "1.0",
            "libigc1_version": "2.0",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        assert not entries

    def test_missing_gpu_device_column(self):
        df = pd.DataFrame([{
            "benchmark": "gemm",
            "compiler": "triton",
            "params": "{}",
            "tflops": 10.0,
            "datetime": "2025-01-01",
        }])
        entries = parse_reports(df, run_id="1", commit_sha="a", tag="ci")
        assert not entries
