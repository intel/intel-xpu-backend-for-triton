"""Tests for benchmark_monitor.model — domain model, enums, parsing helpers, config."""
# pylint: disable=too-few-public-methods,duplicate-code

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from benchmark_monitor.model import (
    AnalysisResult,
    BenchmarkEntry,
    BenchmarkHistory,
    DetectionConfig,
    GpuPlatform,
    MetricResult,
    ThresholdConfig,
    detect_platform,
    parse_benchmark_name,
    parse_params,
)

# ===================================================================
# 1. GpuPlatform enum
# ===================================================================


class TestGpuPlatform:

    def test_pvc_value(self):
        assert GpuPlatform.PVC.value == "pvc"

    def test_bmg_value(self):
        assert GpuPlatform.BMG.value == "bmg"

    def test_str_mixin(self):
        assert str(GpuPlatform.PVC) == "GpuPlatform.PVC"
        assert GpuPlatform.PVC == "pvc"


# ===================================================================
# 2. detect_platform
# ===================================================================


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


# ===================================================================
# 3. parse_benchmark_name
# ===================================================================


class TestParseBenchmarkName:

    def test_typical_key(self):
        assert parse_benchmark_name("gemm/triton/{M:1024,N:1024}") == "gemm"

    def test_no_slash(self):
        assert parse_benchmark_name("softmax") == "softmax"

    def test_multiple_slashes(self):
        assert parse_benchmark_name("flash-attn/triton/{B:2,H:8}") == "flash-attn"

    def test_empty_string(self):
        assert parse_benchmark_name("") == ""


# ===================================================================
# 4. parse_params
# ===================================================================


class TestParseParams:

    def test_typical_key(self):
        assert parse_params("gemm/triton/{M:1024,N:1024}") == "{M:1024,N:1024}"

    def test_no_second_slash(self):
        assert parse_params("gemm/triton") == ""

    def test_no_slash_at_all(self):
        assert parse_params("gemm") == ""

    def test_empty_params(self):
        assert parse_params("gemm/triton/") == ""

    def test_complex_params(self):
        assert parse_params("flash-attn/triton/{B:2,H:8,N_CTX:1024}") == "{B:2,H:8,N_CTX:1024}"


# ===================================================================
# 5. ThresholdConfig defaults match thresholds.yaml
# ===================================================================


class TestThresholdConfig:

    def test_defaults_match_thresholds_yaml(self):
        tc = ThresholdConfig()
        assert tc.min_history == 8
        assert tc.rolling_window == 20
        assert tc.z_threshold == 3.0
        assert tc.min_drop_pct == 5.0
        assert tc.improvement_lock_pct == 8.0
        assert tc.max_cv == 0.15

    def test_custom(self):
        tc = ThresholdConfig(min_history=3, rolling_window=10, z_threshold=2.5, min_drop_pct=10.0)
        assert tc.min_history == 3
        assert tc.rolling_window == 10

    def test_yaml_file_alignment(self):
        """Verify defaults match the actual thresholds.yaml file."""
        yaml_path = Path(__file__).resolve().parent.parent / "thresholds.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            defaults = data.get("defaults", {})
            tc = ThresholdConfig()
            assert tc.min_history == defaults["min_history"]
            assert tc.rolling_window == defaults["rolling_window"]
            assert tc.z_threshold == defaults["z_threshold"]
            assert tc.min_drop_pct == defaults["min_drop_pct"]
            assert tc.improvement_lock_pct == defaults["improvement_lock_pct"]
            assert tc.max_cv == defaults["max_cv"]


# ===================================================================
# 6. DetectionConfig
# ===================================================================


class TestDetectionConfig:

    def test_for_benchmark_default(self):
        cfg = DetectionConfig()
        tc = cfg.for_benchmark("unknown-bench")
        assert tc.min_history == 8

    def test_for_benchmark_override(self):
        override = ThresholdConfig(min_drop_pct=8.0)
        cfg = DetectionConfig(overrides={"softmax": override})
        tc = cfg.for_benchmark("softmax")
        assert tc.min_drop_pct == 8.0

    def test_for_benchmark_miss_returns_defaults(self):
        override = ThresholdConfig(min_drop_pct=8.0)
        cfg = DetectionConfig(overrides={"softmax": override})
        tc = cfg.for_benchmark("gemm")
        assert tc.min_drop_pct == 5.0


# ===================================================================
# 7. Data container construction
# ===================================================================


class TestBenchmarkEntry:

    def test_basic_construction(self):
        entry = BenchmarkEntry(
            run_id="run-1",
            datetime="2025-01-01T00:00:00",
            tag="ci",
            commit_sha="abc123",
            results={"gemm/triton/{M:1024}": {"tflops": 10.0}},
        )
        assert entry.run_id == "run-1"
        assert entry.tag == "ci"
        assert "gemm/triton/{M:1024}" in entry.results

    def test_default_fields(self):
        entry = BenchmarkEntry(run_id="1", datetime="", tag="ci", commit_sha="x")
        assert entry.agama_version == ""
        assert entry.libigc1_version == ""
        assert not entry.results


class TestBenchmarkHistory:

    def test_basic_construction(self):
        hist = BenchmarkHistory(gpu="pvc", entries=[])
        assert hist.gpu == "pvc"
        assert not hist.entries

    def test_with_entries(self):
        e = BenchmarkEntry(run_id="1", datetime="", tag="ci", commit_sha="x")
        hist = BenchmarkHistory(gpu="bmg", entries=[e])
        assert len(hist.entries) == 1


class TestMetricResult:

    def test_construction(self):
        mr = MetricResult(
            key="gemm/triton/{M:1024}",
            benchmark="gemm",
            params="{M:1024}",
            current_tflops=8.0,
            baseline_median=10.0,
            change_pct=-20.0,
            modified_z=-13.5,
        )
        assert mr.driver_change is None
        assert mr.change_pct == -20.0


class TestAnalysisResult:

    def test_defaults(self):
        ar = AnalysisResult()
        assert not ar.regressions
        assert not ar.improvements
        assert ar.skipped == 0
        assert ar.total_checked == 0
        assert ar.driver_change is None
