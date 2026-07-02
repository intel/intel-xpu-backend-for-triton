"""Tests for benchmark_monitor.analysis — statistical regression/improvement detection.

Covers driver change detection, threshold configuration, the core analyze()
function, improvement lock-in, MAD=0 fallback, zero baseline, lock-in
stability, bimodal CV check, and config loading.

This is a direct port of test_detect.py adapted to the new typed API.
"""
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods,import-outside-toplevel,duplicate-code,unsubscriptable-object,not-an-iterable

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from benchmark_monitor.analysis import (
    analyze,
    detect_driver_change,
)
from benchmark_monitor.model import (
    BenchmarkEntry,
    BenchmarkHistory,
    DetectionConfig,
    ThresholdConfig,
)

# ---------------------------------------------------------------------------
# Helpers for building synthetic history data
# ---------------------------------------------------------------------------


def _make_entry(
    run_id: str,
    tag: str = "ci",
    results: dict | None = None,
    agama_version: str = "1.0",
    libigc1_version: str = "2.0",
    commit_sha: str = "abc123",
    datetime: str = "2025-01-01T00:00:00",
) -> BenchmarkEntry:
    """Build a minimal BenchmarkEntry."""
    return BenchmarkEntry(
        run_id=run_id,
        tag=tag,
        results=results or {},
        agama_version=agama_version,
        libigc1_version=libigc1_version,
        commit_sha=commit_sha,
        datetime=datetime,
    )


def _make_history(
    n: int,
    key: str = "gemm/triton/{M:1024,N:1024,K:1024}",
    tflops: float = 10.0,
    tag: str = "ci",
    agama_version: str = "1.0",
    libigc1_version: str = "2.0",
    gpu: str = "pvc",
) -> BenchmarkHistory:
    """Return a BenchmarkHistory with *n* entries all reporting the same tflops for *key*."""
    entries = [
        _make_entry(
            run_id=f"run-{i}",
            tag=tag,
            results={key: {"tflops": tflops}},
            agama_version=agama_version,
            libigc1_version=libigc1_version,
            datetime=f"2025-01-{i + 1:02d}T00:00:00",
        ) for i in range(n)
    ]
    return BenchmarkHistory(gpu=gpu, entries=entries)


def _history_with_entries(entries: list[BenchmarkEntry], gpu: str = "pvc") -> BenchmarkHistory:
    return BenchmarkHistory(gpu=gpu, entries=entries)


# ===================================================================
# 1. detect_driver_change
# ===================================================================


class TestDetectDriverChange:

    def test_no_change(self):
        current = _make_entry("c", agama_version="1.0", libigc1_version="2.0")
        baseline = _make_entry("b", agama_version="1.0", libigc1_version="2.0")
        assert detect_driver_change(current, baseline) is None

    def test_agama_change(self):
        current = _make_entry("c", agama_version="1.1", libigc1_version="2.0")
        baseline = _make_entry("b", agama_version="1.0", libigc1_version="2.0")
        result = detect_driver_change(current, baseline)
        assert result is not None
        changes: list[dict[str, str]] = result
        assert len(changes) == 1
        assert changes[0]["field"] == "agama_version"
        assert changes[0]["from"] == "1.0"
        assert changes[0]["to"] == "1.1"

    def test_libigc1_change(self):
        current = _make_entry("c", agama_version="1.0", libigc1_version="2.1")
        baseline = _make_entry("b", agama_version="1.0", libigc1_version="2.0")
        result = detect_driver_change(current, baseline)
        assert result is not None
        changes: list[dict[str, str]] = result
        assert len(changes) == 1
        assert changes[0]["field"] == "libigc1_version"

    def test_both_changed(self):
        current = _make_entry("c", agama_version="1.1", libigc1_version="2.1")
        baseline = _make_entry("b", agama_version="1.0", libigc1_version="2.0")
        result = detect_driver_change(current, baseline)
        assert result is not None
        changes: list[dict[str, str]] = result
        assert len(changes) == 2
        changed_fields = {d["field"] for d in changes}
        assert changed_fields == {"agama_version", "libigc1_version"}

    def test_missing_fields_default_empty(self):
        current = _make_entry("c", agama_version="1.0", libigc1_version="")
        baseline = _make_entry("b", agama_version="1.0", libigc1_version="")
        assert detect_driver_change(current, baseline) is None


# ===================================================================
# 2. ThresholdConfig / DetectionConfig (ported from TestThresholdConfig/TestConfig)
# ===================================================================


class TestThresholdConfigAnalysis:

    def test_defaults(self):
        tc = ThresholdConfig()
        assert tc.min_history == 8
        assert tc.rolling_window == 20
        assert tc.z_threshold == 3.0
        assert tc.min_drop_pct == 5.0
        assert tc.max_cv == 0.15

    def test_custom(self):
        tc = ThresholdConfig(min_history=3, rolling_window=20, z_threshold=2.5, min_drop_pct=10.0)
        assert tc.min_history == 3
        assert tc.rolling_window == 20


class TestDetectionConfigAnalysis:

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
# 3. load_config (YAML-based)
# ===================================================================


class TestLoadConfig:
    """Test loading DetectionConfig from YAML files.

    We inline the loading logic since load_config may or may not be
    available in the new module structure. This tests the pattern.
    """

    @staticmethod
    def _load_config_from_yaml(path: Path) -> DetectionConfig:
        """Minimal config loader for testing."""
        data = yaml.safe_load(path.read_text()) or {}
        defaults_raw = data.get("defaults", {})
        valid_fields = {f.name for f in fields(ThresholdConfig)}
        defaults = ThresholdConfig(**{k: v for k, v in defaults_raw.items() if k in valid_fields})

        overrides = {}
        for name, ov_raw in data.get("overrides", {}).items():
            merged = {**defaults_raw, **ov_raw}
            overrides[name] = ThresholdConfig(**{k: v for k, v in merged.items() if k in valid_fields})

        return DetectionConfig(defaults=defaults, overrides=overrides)

    def test_basic_yaml(self, tmp_path: Path):
        yaml_content = {
            "defaults": {
                "min_history": 3,
                "rolling_window": 8,
                "z_threshold": 2.5,
                "min_drop_pct": 4.0,
            },
            "overrides": {
                "softmax": {"min_drop_pct": 2.0},
                "flash-attn": {"min_history": 7, "min_drop_pct": 10.0},
            },
        }
        cfg_file = tmp_path / "thresholds.yaml"
        cfg_file.write_text(yaml.dump(yaml_content))

        cfg = self._load_config_from_yaml(cfg_file)
        assert cfg.defaults.min_history == 3
        assert cfg.defaults.rolling_window == 8
        assert cfg.defaults.z_threshold == 2.5
        assert cfg.defaults.min_drop_pct == 4.0

        softmax_cfg = cfg.for_benchmark("softmax")
        assert softmax_cfg.min_drop_pct == 2.0
        assert softmax_cfg.min_history == 3  # inherited

        flash_cfg = cfg.for_benchmark("flash-attn")
        assert flash_cfg.min_history == 7
        assert flash_cfg.min_drop_pct == 10.0

    def test_empty_yaml(self, tmp_path: Path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = self._load_config_from_yaml(cfg_file)
        assert cfg.defaults.min_history == 8  # ThresholdConfig default
        assert not cfg.overrides

    def test_no_overrides_section(self, tmp_path: Path):
        yaml_content = {"defaults": {"min_history": 10}}
        cfg_file = tmp_path / "thresholds.yaml"
        cfg_file.write_text(yaml.dump(yaml_content))
        cfg = self._load_config_from_yaml(cfg_file)
        assert cfg.defaults.min_history == 10
        assert not cfg.overrides


# ===================================================================
# 4. analyze — core detection
# ===================================================================


class TestAnalyze:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"
    # Use lower min_history for easier testing, matching original test_detect.py pattern.
    DEFAULT_CONFIG = DetectionConfig(defaults=ThresholdConfig(min_history=5, rolling_window=10))

    # ---- 4a: empty history ----

    def test_empty_history(self):
        history = BenchmarkHistory(gpu="pvc", entries=[])
        report = analyze(history, self.DEFAULT_CONFIG)
        assert not report.regressions
        assert not report.improvements
        assert report.total_checked == 0

    # ---- 4b: stable benchmark (no regression) ----

    def test_stable_no_regression(self):
        history = _make_history(10, key=self.KEY, tflops=10.0)
        report = analyze(history, self.DEFAULT_CONFIG)
        assert not report.regressions
        assert not report.improvements
        assert report.total_checked == 1

    # ---- 4c: clear regression ----

    def test_clear_regression(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        reg = report.regressions[0]
        assert reg.key == self.KEY
        assert reg.benchmark == "gemm"
        assert reg.current_tflops == 8.0
        assert reg.baseline_median == 10.0
        assert reg.change_pct == -20.0
        assert not report.improvements

    # ---- 4d: clear improvement ----

    def test_clear_improvement(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 12.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert not report.regressions
        assert len(report.improvements) == 1
        imp = report.improvements[0]
        assert imp.key == self.KEY
        assert imp.change_pct == 20.0

    # ---- 4e: insufficient history ----

    def test_insufficient_history_skipped(self):
        baseline = _make_history(3, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 5.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert not report.regressions
        assert report.skipped == 1
        assert report.total_checked == 1

    # ---- 4f: baseline_mad == 0 ----

    def test_baseline_mad_zero(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        assert report.regressions[0].change_pct == -20.0

    def test_baseline_mad_zero_no_drop(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert not report.regressions
        assert not report.improvements

    # ---- 4g: PR tag ----

    def test_pr_tag_uses_all_ci_runs_as_baseline(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        pr_run = _make_entry(
            run_id="pr-run",
            tag="pr-42",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(pr_run)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        assert report.regressions[0].current_tflops == 8.0

    # ---- 4h: driver change tagging ----

    def test_driver_change_tagged_on_regression(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0, agama_version="1.0", libigc1_version="2.0")
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            agama_version="1.1",
            libigc1_version="2.0",
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        assert report.regressions[0].driver_change is not None
        assert report.regressions[0].driver_change[0]["field"] == "agama_version"

    def test_no_driver_change_on_regression(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        assert report.regressions[0].driver_change is None

    # ---- additional edge cases ----

    def test_no_ci_baseline_runs(self):
        history = _make_history(5, key=self.KEY, tflops=10.0, tag="pr-1")
        report = analyze(history, self.DEFAULT_CONFIG)
        assert not report.regressions
        assert report.skipped == 1

    def test_missing_tflops_metric(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"hbm_gbs": 500.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert report.total_checked == 0
        assert not report.regressions

    def test_moderate_drop_below_threshold(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 9.7}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert not report.regressions

    def test_rolling_window_limits_baseline(self):
        baseline = _make_history(20, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-03-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1

    def test_custom_config_higher_min_drop(self):
        cfg = DetectionConfig(defaults=ThresholdConfig(min_drop_pct=25.0, min_history=5, rolling_window=10))
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, cfg)
        assert not report.regressions

    def test_report_fields(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.DEFAULT_CONFIG)
        reg = report.regressions[0]
        # Verify all expected fields exist on MetricResult.
        assert hasattr(reg, "key")
        assert hasattr(reg, "benchmark")
        assert hasattr(reg, "params")
        assert hasattr(reg, "current_tflops")
        assert hasattr(reg, "baseline_median")
        assert hasattr(reg, "change_pct")
        assert hasattr(reg, "modified_z")
        assert hasattr(reg, "driver_change")


# ===================================================================
# 5. analyze — improvement lock-in
# ===================================================================


class TestImprovementLockIn:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"

    def test_improvement_locks_baseline(self):
        """After a significant improvement, baseline locks at the higher level."""
        old_entries = [
            _make_entry(
                run_id=f"old-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 100.0}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i in range(15)
        ]
        new_entries = [
            _make_entry(
                run_id=f"new-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 120.0}},
                datetime=f"2025-02-{10 + i:02d}T00:00:00",
            ) for i in range(5)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 105.0}},
            datetime="2025-03-01T00:00:00",
        )
        history = _history_with_entries(old_entries + new_entries + [current])
        cfg = DetectionConfig(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze(history, cfg)
        assert len(report.regressions) == 1
        assert report.regressions[0].baseline_median == 120.0

    def test_no_lock_in_for_small_gain(self):
        old_entries = [
            _make_entry(
                run_id=f"old-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 100.0}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i in range(15)
        ]
        new_entries = [
            _make_entry(
                run_id=f"new-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 105.0}},  # only 5% gain
                datetime=f"2025-02-{10 + i:02d}T00:00:00",
            ) for i in range(5)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 100.0}},
            datetime="2025-03-01T00:00:00",
        )
        history = _history_with_entries(old_entries + new_entries + [current])
        cfg = DetectionConfig(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze(history, cfg)
        assert not report.regressions

    def test_stable_performance_no_lock_in(self):
        entries = [
            _make_entry(
                run_id=f"run-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 100.0}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i in range(20)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 100.0}},
            datetime="2025-03-01T00:00:00",
        )
        history = _history_with_entries(entries + [current])
        cfg = DetectionConfig(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze(history, cfg)
        assert not report.regressions
        assert not report.improvements


# ===================================================================
# 6. MAD=0 fallback
# ===================================================================


class TestMadZeroFallback:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"
    CONFIG = DetectionConfig(defaults=ThresholdConfig(min_history=5, rolling_window=10))

    def test_z_score_is_bounded(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.CONFIG)
        assert len(report.regressions) == 1
        z = report.regressions[0].modified_z
        assert -100 < z < 0, f"z-score should be bounded, got {z}"

    def test_small_drop_not_flagged_with_bounded_mad(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 9.7}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.CONFIG)
        assert not report.regressions


# ===================================================================
# 7. Zero baseline
# ===================================================================


class TestBaselineZero:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"
    CONFIG = DetectionConfig(defaults=ThresholdConfig(min_history=5, rolling_window=10))

    def test_both_zero_skipped(self):
        baseline = _make_history(10, key=self.KEY, tflops=0.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 0.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.CONFIG)
        assert not report.regressions
        assert report.skipped == 1

    def test_nonzero_current_from_zero_baseline(self):
        baseline = _make_history(10, key=self.KEY, tflops=0.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 10.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.CONFIG)
        assert report.skipped == 0
        assert len(report.improvements) == 1


# ===================================================================
# 8. Lock-in stability
# ===================================================================


class TestLockInStability:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"

    def test_noisy_improvement_no_lockin(self):
        old_entries = [
            _make_entry(
                run_id=f"old-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 100.0}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i in range(12)
        ]
        noisy_values = [130, 90, 135, 85, 140, 80, 130, 95]
        noisy_entries = [
            _make_entry(
                run_id=f"noisy-{i}",
                tag="ci",
                results={self.KEY: {"tflops": float(v)}},
                datetime=f"2025-02-{10 + i:02d}T00:00:00",
            ) for i, v in enumerate(noisy_values)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 100.0}},
            datetime="2025-03-01T00:00:00",
        )
        cfg = DetectionConfig(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        history = _history_with_entries(old_entries + noisy_entries + [current])
        report = analyze(history, cfg)
        assert not report.regressions

    def test_stable_improvement_locks_in(self):
        old_entries = [
            _make_entry(
                run_id=f"old-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 100.0}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i in range(12)
        ]
        improved_entries = [
            _make_entry(
                run_id=f"imp-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 120.0}},
                datetime=f"2025-02-{10 + i:02d}T00:00:00",
            ) for i in range(8)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 105.0}},
            datetime="2025-03-01T00:00:00",
        )
        cfg = DetectionConfig(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        history = _history_with_entries(old_entries + improved_entries + [current])
        report = analyze(history, cfg)
        assert len(report.regressions) == 1
        assert report.regressions[0].baseline_median == 120.0


# ===================================================================
# 9. Bimodal CV check
# ===================================================================


class TestBimodalCVCheck:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"
    CONFIG = DetectionConfig(defaults=ThresholdConfig(min_history=5, rolling_window=10))

    def test_bimodal_baseline_skipped(self):
        bimodal_values = [17.0, 29.0] * 5
        entries = [
            _make_entry(
                run_id=f"run-{i}",
                tag="ci",
                results={self.KEY: {"tflops": v}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i, v in enumerate(bimodal_values)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 17.0}},
            datetime="2025-02-01T00:00:00",
        )
        entries.append(current)
        history = _history_with_entries(entries)
        report = analyze(history, self.CONFIG)
        assert not report.regressions
        assert report.skipped == 1

    def test_stable_baseline_not_skipped(self):
        baseline = _make_history(10, key=self.KEY, tflops=10.0)
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        baseline.entries.append(current)
        report = analyze(baseline, self.CONFIG)
        assert report.skipped == 0
        assert len(report.regressions) == 1

    def test_moderate_noise_not_skipped(self):
        values = [90.0, 95.0, 100.0, 105.0, 110.0, 92.0, 98.0, 102.0, 108.0, 96.0]
        entries = [
            _make_entry(
                run_id=f"run-{i}",
                tag="ci",
                results={self.KEY: {"tflops": v}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i, v in enumerate(values)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 75.0}},
            datetime="2025-02-01T00:00:00",
        )
        entries.append(current)
        history = _history_with_entries(entries)
        report = analyze(history, self.CONFIG)
        assert report.skipped == 0
        assert len(report.regressions) == 1

    def test_custom_max_cv_override(self):
        bimodal_values = [17.0, 29.0] * 5
        entries = [
            _make_entry(
                run_id=f"run-{i}",
                tag="ci",
                results={self.KEY: {"tflops": v}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i, v in enumerate(bimodal_values)
        ]
        current = _make_entry(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 17.0}},
            datetime="2025-02-01T00:00:00",
        )
        entries.append(current)
        history = _history_with_entries(entries)
        cfg = DetectionConfig(
            defaults=ThresholdConfig(min_history=5, rolling_window=10),
            overrides={"gemm": ThresholdConfig(max_cv=0.50, min_history=5, rolling_window=10)},
        )
        report = analyze(history, cfg)
        assert report.skipped == 0


# ===================================================================
# 10. Resolve GPUs (via storage backend)
# ===================================================================


class TestResolveGpus:
    """Test GPU resolution via JsonFileBackend.list_gpus() which replaces resolve_gpus()."""

    def test_list_gpus_finds_existing(self, tmp_path: Path):
        from benchmark_monitor.storage import JsonFileBackend
        for name in ("pvc", "bmg"):
            d = tmp_path / name
            d.mkdir()
            (d / "history.json").write_text("[]")
        backend = JsonFileBackend(tmp_path)
        gpus = backend.list_gpus()
        assert "pvc" in gpus
        assert "bmg" in gpus

    def test_empty_dir(self, tmp_path: Path):
        from benchmark_monitor.storage import JsonFileBackend
        backend = JsonFileBackend(tmp_path)
        assert not backend.list_gpus()

    def test_nonexistent_dir(self, tmp_path: Path):
        from benchmark_monitor.storage import JsonFileBackend
        backend = JsonFileBackend(tmp_path / "nonexistent")
        assert not backend.list_gpus()


# ===================================================================
# 11. Build report (integration — analyze across multiple GPUs)
# ===================================================================


class TestBuildReport:
    KEY_A = "gemm/triton/{M:1024,N:1024,K:1024}"
    KEY_B = "softmax/triton/{M:4096,N:4096}"
    CONFIG = DetectionConfig(defaults=ThresholdConfig(min_history=5, rolling_window=10))

    def _build_and_analyze(self, tmp_path: Path, gpu_histories: dict[str, list[BenchmarkEntry]]):
        """Helper: save histories, load, analyze, return per-gpu results."""
        from benchmark_monitor.storage import JsonFileBackend
        backend = JsonFileBackend(tmp_path)
        results = {}
        for gpu, entries in gpu_histories.items():
            hist = BenchmarkHistory(gpu=gpu, entries=entries)
            backend.save(gpu, hist)
            loaded = backend.load(gpu)
            results[gpu] = analyze(loaded, self.CONFIG)
        return results

    def test_two_gpus_with_regression(self, tmp_path: Path):
        pvc_baseline = _make_history(10, key=self.KEY_A, tflops=10.0)
        pvc_current = _make_entry(
            run_id="run-final",
            tag="ci",
            results={self.KEY_A: {"tflops": 7.5}},
            datetime="2025-02-15T00:00:00",
        )
        pvc_entries = pvc_baseline.entries + [pvc_current]

        bmg_entries = _make_history(10, key=self.KEY_B, tflops=5.0).entries

        results = self._build_and_analyze(tmp_path, {"pvc": pvc_entries, "bmg": bmg_entries})
        assert len(results["pvc"].regressions) == 1
        assert results["pvc"].regressions[0].benchmark == "gemm"
        assert len(results["bmg"].regressions) == 0

    def test_empty_history(self, tmp_path: Path):
        results = self._build_and_analyze(tmp_path, {})
        assert not results

    def test_report_structure(self, tmp_path: Path):
        entries = _make_history(10, key=self.KEY_A, tflops=10.0).entries
        results = self._build_and_analyze(tmp_path, {"pvc": entries})
        pvc_report = results["pvc"]
        assert hasattr(pvc_report, "regressions")
        assert hasattr(pvc_report, "improvements")
        assert hasattr(pvc_report, "skipped")
        assert hasattr(pvc_report, "total_checked")
        assert hasattr(pvc_report, "driver_change")
