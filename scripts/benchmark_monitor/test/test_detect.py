"""Tests for benchmark regression detection (detect_regressions.py).

Covers key-parsing helpers, driver change detection, threshold configuration,
statistical analysis (analyze_gpu), GPU resolution, and the top-level
build_report integration path.
"""
# pylint: disable=too-many-arguments,too-many-positional-arguments,use-implicit-booleaness-not-comparison

from __future__ import annotations

import json
from pathlib import Path

import yaml

from benchmark_monitor.detect_regressions import (
    Config,
    ThresholdConfig,
    analyze_gpu,
    build_report,
    detect_driver_change,
    load_config,
    parse_benchmark_name,
    parse_params,
    resolve_gpus,
)

# ---------------------------------------------------------------------------
# Helpers for building synthetic history data
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str,
    tag: str = "ci",
    results: dict | None = None,
    agama_version: str = "1.0",
    libigc1_version: str = "2.0",
    commit_sha: str = "abc123",
    datetime: str = "2025-01-01T00:00:00",
) -> dict:
    """Build a minimal history-run dict."""
    return {
        "run_id": run_id,
        "tag": tag,
        "results": results or {},
        "agama_version": agama_version,
        "libigc1_version": libigc1_version,
        "commit_sha": commit_sha,
        "datetime": datetime,
    }


def _make_stable_history(
    n: int,
    key: str = "gemm/triton/{M:1024,N:1024,K:1024}",
    tflops: float = 10.0,
    tag: str = "ci",
    agama_version: str = "1.0",
    libigc1_version: str = "2.0",
) -> list[dict]:
    """Return *n* history runs all reporting the same tflops for *key*."""
    return [
        _make_run(
            run_id=f"run-{i}",
            tag=tag,
            results={key: {"tflops": tflops}},
            agama_version=agama_version,
            libigc1_version=libigc1_version,
            datetime=f"2025-01-{i + 1:02d}T00:00:00",
        ) for i in range(n)
    ]


# ===================================================================
# 1. parse_benchmark_name
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
# 2. parse_params
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
# 3. detect_driver_change
# ===================================================================


class TestDetectDriverChange:

    def test_no_change(self):
        current = {"agama_version": "1.0", "libigc1_version": "2.0"}
        baseline = {"agama_version": "1.0", "libigc1_version": "2.0"}
        assert detect_driver_change(current, baseline) is None

    def test_agama_change(self):
        current = {"agama_version": "1.1", "libigc1_version": "2.0"}
        baseline = {"agama_version": "1.0", "libigc1_version": "2.0"}
        result = detect_driver_change(current, baseline)
        assert result is not None
        assert len(result) == 1
        assert result[0]["field"] == "agama_version"
        assert result[0]["from"] == "1.0"
        assert result[0]["to"] == "1.1"

    def test_libigc1_change(self):
        current = {"agama_version": "1.0", "libigc1_version": "2.1"}
        baseline = {"agama_version": "1.0", "libigc1_version": "2.0"}
        result = detect_driver_change(current, baseline)
        assert result is not None
        assert len(result) == 1
        assert result[0]["field"] == "libigc1_version"
        assert result[0]["from"] == "2.0"
        assert result[0]["to"] == "2.1"

    def test_both_changed(self):
        current = {"agama_version": "1.1", "libigc1_version": "2.1"}
        baseline = {"agama_version": "1.0", "libigc1_version": "2.0"}
        result = detect_driver_change(current, baseline)
        # When both change, result is a list of two dicts.
        assert isinstance(result, list)
        assert len(result) == 2
        fields = {d["field"] for d in result}
        assert fields == {"agama_version", "libigc1_version"}

    def test_missing_fields(self):
        current = {"agama_version": "1.0"}
        baseline = {}
        # Missing fields default to "" via .get(); "" != "1.0" triggers change.
        result = detect_driver_change(current, baseline)
        assert result is not None


# ===================================================================
# 4. ThresholdConfig / Config
# ===================================================================


class TestThresholdConfig:

    def test_defaults(self):
        tc = ThresholdConfig()
        assert tc.min_history == 5
        assert tc.rolling_window == 10
        assert tc.z_threshold == 3.0
        assert tc.min_drop_pct == 5.0
        assert tc.max_cv == 0.15

    def test_custom(self):
        tc = ThresholdConfig(min_history=3, rolling_window=20, z_threshold=2.5, min_drop_pct=10.0)
        assert tc.min_history == 3
        assert tc.rolling_window == 20


class TestConfig:

    def test_for_benchmark_default(self):
        cfg = Config()
        tc = cfg.for_benchmark("unknown-bench")
        assert tc.min_history == 5

    def test_for_benchmark_override(self):
        override = ThresholdConfig(min_drop_pct=8.0)
        cfg = Config(overrides={"softmax": override})
        tc = cfg.for_benchmark("softmax")
        assert tc.min_drop_pct == 8.0

    def test_for_benchmark_miss_returns_defaults(self):
        override = ThresholdConfig(min_drop_pct=8.0)
        cfg = Config(overrides={"softmax": override})
        tc = cfg.for_benchmark("gemm")
        assert tc.min_drop_pct == 5.0


# ===================================================================
# 5. load_config
# ===================================================================


class TestLoadConfig:

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

        cfg = load_config(cfg_file)
        assert cfg.defaults.min_history == 3
        assert cfg.defaults.rolling_window == 8
        assert cfg.defaults.z_threshold == 2.5
        assert cfg.defaults.min_drop_pct == 4.0

        # Override inherits defaults for missing fields.
        softmax_cfg = cfg.for_benchmark("softmax")
        assert softmax_cfg.min_drop_pct == 2.0
        assert softmax_cfg.min_history == 3  # inherited

        flash_cfg = cfg.for_benchmark("flash-attn")
        assert flash_cfg.min_history == 7
        assert flash_cfg.min_drop_pct == 10.0

    def test_empty_yaml(self, tmp_path: Path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(cfg_file)
        # Falls back to ThresholdConfig() defaults.
        assert cfg.defaults.min_history == 5
        assert cfg.overrides == {}

    def test_no_overrides_section(self, tmp_path: Path):
        yaml_content = {"defaults": {"min_history": 10}}
        cfg_file = tmp_path / "thresholds.yaml"
        cfg_file.write_text(yaml.dump(yaml_content))
        cfg = load_config(cfg_file)
        assert cfg.defaults.min_history == 10
        assert cfg.overrides == {}


# ===================================================================
# 6. analyze_gpu
# ===================================================================


class TestAnalyzeGpu:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"
    DEFAULT_CONFIG = Config()

    # ---- 6a: empty history ----

    def test_empty_history(self):
        report = analyze_gpu([], self.DEFAULT_CONFIG)
        assert report.regressions == []
        assert report.improvements == []
        assert report.total_checked == 0

    # ---- 6b: stable benchmark (no regression) ----

    def test_stable_no_regression(self):
        history = _make_stable_history(10, key=self.KEY, tflops=10.0)
        # Last run is the "current" and is also ci-tagged with stable tflops.
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        assert report.regressions == []
        assert report.improvements == []
        assert report.total_checked == 1

    # ---- 6c: clear regression ----

    def test_clear_regression(self):
        # 10 stable runs at 10.0, then current drops to 8.0 (20% drop).
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        reg = report.regressions[0]
        assert reg["key"] == self.KEY
        assert reg["benchmark"] == "gemm"
        assert reg["current_tflops"] == 8.0
        assert reg["baseline_median"] == 10.0
        assert reg["drop_pct"] == -20.0
        assert report.improvements == []

    # ---- 6d: clear improvement ----

    def test_clear_improvement(self):
        # 10 stable runs at 10.0, then current jumps to 12.0 (20% gain).
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 12.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        assert report.regressions == []
        assert len(report.improvements) == 1
        imp = report.improvements[0]
        assert imp["key"] == self.KEY
        assert imp["gain_pct"] == 20.0

    # ---- 6e: insufficient history (< min_history) ----

    def test_insufficient_history_skipped(self):
        # Only 3 CI baseline runs + current = 4 total, but min_history=5.
        history = _make_stable_history(3, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 5.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = history + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        assert report.regressions == []
        assert report.skipped == 1
        assert report.total_checked == 1

    # ---- 6f: baseline_mad == 0 (all identical values) ----

    def test_baseline_mad_zero(self):
        # All baseline values are exactly the same -> MAD = 0 -> uses 1e-9 fallback.
        # With a large enough drop it should still detect regression.
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        # MAD=0 -> 1e-9 fallback -> modified_z will be extremely large negative.
        assert len(report.regressions) == 1
        assert report.regressions[0]["drop_pct"] == -20.0

    def test_baseline_mad_zero_no_drop(self):
        # All identical and current matches -> no regression.
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        # The current run is the last element of the stable history.
        report = analyze_gpu(baseline, self.DEFAULT_CONFIG)
        assert report.regressions == []
        assert report.improvements == []

    # ---- 6g: PR tag ----

    def test_pr_tag_uses_all_ci_runs_as_baseline(self):
        # 10 CI runs, then a PR run with a drop. Baseline should include all 10 CI runs.
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        pr_run = _make_run(
            run_id="pr-run",
            tag="pr-42",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [pr_run]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        # Should detect regression for the PR run.
        assert len(report.regressions) == 1
        assert report.regressions[0]["current_tflops"] == 8.0

    # ---- 6h: driver change tagging ----

    def test_driver_change_tagged_on_regression(self):
        # Baseline with driver v1.0, current with driver v1.1 and performance drop.
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0, agama_version="1.0", libigc1_version="2.0")
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            agama_version="1.1",
            libigc1_version="2.0",
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        assert report.regressions[0]["driver_change"] is not None
        assert report.regressions[0]["driver_change"][0]["field"] == "agama_version"

    def test_no_driver_change_on_regression(self):
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        assert len(report.regressions) == 1
        assert report.regressions[0]["driver_change"] is None

    # ---- additional edge cases ----

    def test_no_ci_baseline_runs(self):
        # All runs are PR-tagged, so no CI baseline exists.
        runs = _make_stable_history(5, key=self.KEY, tflops=10.0, tag="pr-1")
        report = analyze_gpu(runs, self.DEFAULT_CONFIG)
        # Last run is PR, baseline is all CI runs = [], so everything skipped.
        assert report.regressions == []
        assert report.skipped == 1

    def test_missing_tflops_metric(self):
        # Current run has a key but no tflops -> should be silently skipped.
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"hbm_gbs": 500.0}},  # no tflops
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        # The current key has no tflops, so total_checked stays 0.
        assert report.total_checked == 0
        assert report.regressions == []

    def test_moderate_drop_below_threshold(self):
        # 3% drop with tight baseline -> drop_pct < min_drop_pct (5%) -> no regression.
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 9.7}},  # 3% drop
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        assert report.regressions == []

    def test_rolling_window_limits_baseline(self):
        # 20 CI runs, but rolling_window defaults to 10. Baseline uses last 10.
        all_runs = _make_stable_history(20, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-03-01T00:00:00",
        )
        history = all_runs + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        # Should still detect regression using the last 10 CI runs as baseline.
        assert len(report.regressions) == 1

    def test_custom_config_higher_min_drop(self):
        # Use config with min_drop_pct=25%. A 20% drop should NOT be flagged.
        cfg = Config(defaults=ThresholdConfig(min_drop_pct=25.0))
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, cfg)
        assert report.regressions == []

    def test_report_fields(self):
        # Verify that regression dict contains all expected fields.
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, self.DEFAULT_CONFIG)
        reg = report.regressions[0]
        expected_keys = {
            "key",
            "benchmark",
            "params",
            "current_tflops",
            "baseline_median",
            "drop_pct",
            "modified_z",
            "driver_change",
        }
        assert set(reg.keys()) == expected_keys


# ===================================================================
# 6.5. analyze_gpu — improvement lock-in
# ===================================================================


class TestImprovementLockIn:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"

    def test_improvement_locks_baseline(self):
        """After a significant improvement, baseline locks at the higher level.

        15 runs at 100, then 5 runs at 120 (+20%). Current at 105 should be
        flagged as a regression relative to the locked-in baseline of 120,
        not the full-window median (~105).
        """
        old_runs = _make_stable_history(15, key=self.KEY, tflops=100.0)
        new_runs = [
            _make_run(
                run_id=f"new-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 120.0}},
                datetime=f"2025-02-{10+i:02d}T00:00:00",
            ) for i in range(5)
        ]
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 105.0}},
            datetime="2025-03-01T00:00:00",
        )
        history = old_runs + new_runs + [current]
        # Use config with rolling_window=20 so all 20 baseline runs are included.
        cfg = Config(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze_gpu(history, cfg)
        # The recent median (120) is >10% above the full median (~105),
        # so the baseline locks at 120. Current 105 is a ~12.5% drop -> regression.
        assert len(report.regressions) == 1
        assert report.regressions[0]["baseline_median"] == 120.0

    def test_no_lock_in_for_small_gain(self):
        """A small improvement (<10%) does not trigger lock-in."""
        old_runs = _make_stable_history(15, key=self.KEY, tflops=100.0)
        new_runs = [
            _make_run(
                run_id=f"new-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 105.0}},  # only 5% gain
                datetime=f"2025-02-{10+i:02d}T00:00:00",
            ) for i in range(5)
        ]
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 100.0}},
            datetime="2025-03-01T00:00:00",
        )
        history = old_runs + new_runs + [current]
        cfg = Config(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze_gpu(history, cfg)
        # 5% gain < 10% threshold -> no lock-in -> full window median ~101.25.
        # Current at 100 is close to the full median -> no regression.
        assert report.regressions == []

    def test_stable_performance_no_lock_in(self):
        """When performance is stable, full window is used (more robust)."""
        history = _make_stable_history(20, key=self.KEY, tflops=100.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 100.0}},
            datetime="2025-03-01T00:00:00",
        )
        history = history + [current]
        cfg = Config(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze_gpu(history, cfg)
        assert report.regressions == []
        assert report.improvements == []


# ===================================================================
# 7. resolve_gpus
# ===================================================================


class TestResolveGpus:

    def test_known_runner_label(self, tmp_path: Path):
        gpu_dir = tmp_path / "pvc"
        gpu_dir.mkdir()
        (gpu_dir / "history.json").write_text("[]")
        result = resolve_gpus("max1550", tmp_path)
        assert "pvc" in result
        assert result["pvc"] == gpu_dir / "history.json"

    def test_known_runner_label_bmg(self, tmp_path: Path):
        gpu_dir = tmp_path / "bmg"
        gpu_dir.mkdir()
        (gpu_dir / "history.json").write_text("[]")
        result = resolve_gpus("b580", tmp_path)
        assert "bmg" in result

    def test_unknown_runner_label_scans_all(self, tmp_path: Path):
        for name in ("pvc", "bmg"):
            d = tmp_path / name
            d.mkdir()
            (d / "history.json").write_text("[]")
        result = resolve_gpus("unknown-runner", tmp_path)
        assert "pvc" in result
        assert "bmg" in result

    def test_empty_label_scans_all(self, tmp_path: Path):
        for name in ("pvc", "bmg"):
            d = tmp_path / name
            d.mkdir()
            (d / "history.json").write_text("[]")
        result = resolve_gpus("", tmp_path)
        assert "pvc" in result
        assert "bmg" in result

    def test_missing_history_file(self, tmp_path: Path):
        # Directory exists but no history.json.
        (tmp_path / "pvc").mkdir()
        result = resolve_gpus("max1550", tmp_path)
        assert result == {}

    def test_nonexistent_history_dir(self, tmp_path: Path):
        result = resolve_gpus("", tmp_path / "nonexistent")
        assert result == {}

    def test_scan_ignores_files(self, tmp_path: Path):
        # A regular file in the history dir should not be picked up.
        (tmp_path / "not-a-dir.txt").write_text("junk")
        d = tmp_path / "pvc"
        d.mkdir()
        (d / "history.json").write_text("[]")
        result = resolve_gpus("", tmp_path)
        assert list(result.keys()) == ["pvc"]


# ===================================================================
# 8. build_report (integration)
# ===================================================================


class TestBuildReport:
    KEY_A = "gemm/triton/{M:1024,N:1024,K:1024}"
    KEY_B = "softmax/triton/{M:4096,N:4096}"

    def _write_history(self, path: Path, history: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history))

    def test_two_gpus_with_regression(self, tmp_path: Path):
        cfg = Config()

        # PVC: stable history + regression on current run.
        pvc_baseline = _make_stable_history(10, key=self.KEY_A, tflops=10.0)
        pvc_current = _make_run(
            run_id="run-final",
            tag="ci",
            results={self.KEY_A: {"tflops": 7.5}},
            datetime="2025-02-15T00:00:00",
        )
        pvc_history = pvc_baseline + [pvc_current]
        self._write_history(tmp_path / "pvc" / "history.json", pvc_history)

        # BMG: stable history, no regression.
        bmg_history = _make_stable_history(10, key=self.KEY_B, tflops=5.0)
        self._write_history(tmp_path / "bmg" / "history.json", bmg_history)

        report = build_report(tmp_path, runner_label="", config=cfg)

        # Top-level metadata comes from the latest run across GPUs.
        assert report["run_id"] == "run-final"
        assert "gpus" in report

        # PVC should have 1 regression.
        pvc_data = report["gpus"]["pvc"]
        assert len(pvc_data["regressions"]) == 1
        assert pvc_data["regressions"][0]["benchmark"] == "gemm"

        # BMG should have no regressions.
        bmg_data = report["gpus"]["bmg"]
        assert len(bmg_data["regressions"]) == 0

    def test_single_gpu_via_runner_label(self, tmp_path: Path):
        cfg = Config()

        pvc_history = _make_stable_history(10, key=self.KEY_A, tflops=10.0)
        self._write_history(tmp_path / "pvc" / "history.json", pvc_history)

        bmg_history = _make_stable_history(10, key=self.KEY_B, tflops=5.0)
        self._write_history(tmp_path / "bmg" / "history.json", bmg_history)

        report = build_report(tmp_path, runner_label="max1550", config=cfg)
        # Only PVC should be in the report.
        assert "pvc" in report["gpus"]
        assert "bmg" not in report["gpus"]

    def test_empty_history_dir(self, tmp_path: Path):
        cfg = Config()
        report = build_report(tmp_path, runner_label="", config=cfg)
        assert report["gpus"] == {}
        assert report["run_id"] == ""

    def test_report_structure(self, tmp_path: Path):
        cfg = Config()
        history = _make_stable_history(10, key=self.KEY_A, tflops=10.0)
        self._write_history(tmp_path / "pvc" / "history.json", history)

        report = build_report(tmp_path, runner_label="", config=cfg)
        # Verify top-level keys.
        assert set(report.keys()) == {"run_id", "datetime", "commit_sha", "gpus"}
        # Verify per-GPU keys.
        gpu_data = report["gpus"]["pvc"]
        assert set(gpu_data.keys()) == {
            "regressions",
            "improvements",
            "skipped",
            "total_checked",
            "driver_change",
        }


# ===================================================================
# MAD=0 fallback: perfectly stable baselines produce interpretable z-scores
# ===================================================================


class TestMadZeroFallback:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"

    def test_z_score_is_bounded(self):
        """When baseline is perfectly stable (MAD=0), z-score should be reasonable."""
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        report = analyze_gpu(baseline + [current], Config())
        assert len(report.regressions) == 1
        z = report.regressions[0]["modified_z"]
        # With 1% MAD fallback: MAD=0.1, z = 0.6745*(8-10)/0.1 = -13.5
        assert -100 < z < 0, f"z-score should be bounded, got {z}"

    def test_small_drop_not_flagged_with_bounded_mad(self):
        """A 3% drop on stable baseline should NOT be flagged (below min_drop_pct=5%)."""
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 9.7}},
            datetime="2025-02-01T00:00:00",
        )
        report = analyze_gpu(baseline + [current], Config())
        assert report.regressions == []


# ===================================================================
# Zero baseline: detection when baseline_median is zero
# ===================================================================


class TestBaselineZero:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"

    def test_both_zero_skipped(self):
        """When both baseline and current are zero, should be skipped."""
        baseline = _make_stable_history(10, key=self.KEY, tflops=0.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 0.0}},
            datetime="2025-02-01T00:00:00",
        )
        report = analyze_gpu(baseline + [current], Config())
        assert report.regressions == []
        assert report.skipped == 1

    def test_nonzero_current_from_zero_baseline(self):
        """Non-zero current from zero baseline should be detected as improvement, not skipped."""
        baseline = _make_stable_history(10, key=self.KEY, tflops=0.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 10.0}},
            datetime="2025-02-01T00:00:00",
        )
        report = analyze_gpu(baseline + [current], Config())
        # Should NOT be permanently skipped -- should detect as improvement
        assert report.skipped == 0
        assert len(report.improvements) == 1


# ===================================================================
# Lock-in stability: improvement lock-in requires low variability
# ===================================================================


class TestLockInStability:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"

    def test_noisy_improvement_no_lockin(self):
        """Noisy recent values should NOT trigger lock-in even if median is high."""
        old_runs = _make_stable_history(12, key=self.KEY, tflops=100.0)
        # Last 8 runs alternate wildly: high variability
        noisy_values = [130, 90, 135, 85, 140, 80, 130, 95]
        noisy_runs = [
            _make_run(
                run_id=f"noisy-{i}",
                tag="ci",
                results={self.KEY: {"tflops": v}},
                datetime=f"2025-02-{10+i:02d}T00:00:00",
            ) for i, v in enumerate(noisy_values)
        ]
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 100.0}},
            datetime="2025-03-01T00:00:00",
        )
        cfg = Config(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze_gpu(old_runs + noisy_runs + [current], cfg)
        # 100 should NOT be flagged as regression vs noisy baseline
        assert report.regressions == []

    def test_stable_improvement_locks_in(self):
        """Stable, significant improvement should trigger lock-in."""
        old_runs = _make_stable_history(12, key=self.KEY, tflops=100.0)
        # Last 8 runs are consistently at 120 (stable improvement)
        improved_runs = [
            _make_run(
                run_id=f"imp-{i}",
                tag="ci",
                results={self.KEY: {"tflops": 120.0}},
                datetime=f"2025-02-{10+i:02d}T00:00:00",
            ) for i in range(8)
        ]
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 105.0}},
            datetime="2025-03-01T00:00:00",
        )
        cfg = Config(defaults=ThresholdConfig(rolling_window=20, min_history=5))
        report = analyze_gpu(old_runs + improved_runs + [current], cfg)
        # Baseline should lock at 120. Drop from 120 to 105 is ~12.5% -> regression
        assert len(report.regressions) == 1
        assert report.regressions[0]["baseline_median"] == 120.0


# ===================================================================
# Bimodal CV check: skip metrics with high coefficient of variation
# ===================================================================


class TestBimodalCVCheck:
    KEY = "gemm/triton/{M:1024,N:1024,K:1024}"

    def test_bimodal_baseline_skipped(self):
        """Bimodal baseline (CV ~0.30) should be skipped."""
        bimodal_values = [17.0, 29.0] * 5
        history = [
            _make_run(
                run_id=f"run-{i}",
                tag="ci",
                results={self.KEY: {"tflops": v}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i, v in enumerate(bimodal_values)
        ]
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 17.0}},
            datetime="2025-02-01T00:00:00",
        )
        history.append(current)
        report = analyze_gpu(history, Config())
        assert report.regressions == []
        assert report.skipped == 1

    def test_stable_baseline_not_skipped(self):
        """Stable baseline (CV=0) should not be skipped; regression detected."""
        baseline = _make_stable_history(10, key=self.KEY, tflops=10.0)
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 8.0}},
            datetime="2025-02-01T00:00:00",
        )
        history = baseline + [current]
        report = analyze_gpu(history, Config())
        assert report.skipped == 0
        assert len(report.regressions) == 1

    def test_moderate_noise_not_skipped(self):
        """Moderate noise (CV ~0.07) should not be skipped; regression detected."""
        values = [90.0, 95.0, 100.0, 105.0, 110.0, 92.0, 98.0, 102.0, 108.0, 96.0]
        history = [
            _make_run(
                run_id=f"run-{i}",
                tag="ci",
                results={self.KEY: {"tflops": v}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i, v in enumerate(values)
        ]
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 75.0}},
            datetime="2025-02-01T00:00:00",
        )
        history.append(current)
        report = analyze_gpu(history, Config())
        assert report.skipped == 0
        assert len(report.regressions) == 1

    def test_custom_max_cv_override(self):
        """Per-benchmark max_cv override allows higher CV without skipping."""
        bimodal_values = [17.0, 29.0] * 5
        history = [
            _make_run(
                run_id=f"run-{i}",
                tag="ci",
                results={self.KEY: {"tflops": v}},
                datetime=f"2025-01-{i + 1:02d}T00:00:00",
            ) for i, v in enumerate(bimodal_values)
        ]
        current = _make_run(
            run_id="current",
            tag="ci",
            results={self.KEY: {"tflops": 17.0}},
            datetime="2025-02-01T00:00:00",
        )
        history.append(current)
        # Override max_cv to 0.50 so CV ~0.30 is within tolerance.
        cfg = Config(overrides={"gemm": ThresholdConfig(max_cv=0.50)})
        report = analyze_gpu(history, cfg)
        assert report.skipped == 0
