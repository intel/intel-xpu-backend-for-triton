"""Detect performance regressions in benchmark history using Modified Z-Score.

Reads historical benchmark data from JSON files, computes rolling baselines
from CI-tagged runs, and flags regressions/improvements that exceed
configurable thresholds.
"""
# pylint: disable=too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ThresholdConfig:
    """Per-benchmark threshold settings."""

    min_history: int = 5
    rolling_window: int = 10
    z_threshold: float = 3.0
    min_drop_pct: float = 5.0
    improvement_lock_pct: float = 10.0
    max_cv: float = 0.15


@dataclass
class Config:
    """Full threshold configuration with per-benchmark overrides."""

    defaults: ThresholdConfig = field(default_factory=ThresholdConfig)
    overrides: dict[str, ThresholdConfig] = field(default_factory=dict)

    def for_benchmark(self, benchmark: str) -> ThresholdConfig:
        return self.overrides.get(benchmark, self.defaults)


def load_config(path: Path) -> Config:
    """Load thresholds.yaml and return a Config."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults_raw = raw.get("defaults", {})
    defaults = ThresholdConfig(
        min_history=defaults_raw.get("min_history", 5),
        rolling_window=defaults_raw.get("rolling_window", 10),
        z_threshold=defaults_raw.get("z_threshold", 3.0),
        min_drop_pct=defaults_raw.get("min_drop_pct", 5.0),
        improvement_lock_pct=defaults_raw.get("improvement_lock_pct", 10.0),
        max_cv=defaults_raw.get("max_cv", 0.15),
    )

    overrides: dict[str, ThresholdConfig] = {}
    for name, vals in raw.get("overrides", {}).items():
        overrides[name] = ThresholdConfig(
            min_history=vals.get("min_history", defaults.min_history),
            rolling_window=vals.get("rolling_window", defaults.rolling_window),
            z_threshold=vals.get("z_threshold", defaults.z_threshold),
            min_drop_pct=vals.get("min_drop_pct", defaults.min_drop_pct),
            improvement_lock_pct=vals.get("improvement_lock_pct", defaults.improvement_lock_pct),
            max_cv=vals.get("max_cv", defaults.max_cv),
        )

    return Config(defaults=defaults, overrides=overrides)


# ---------------------------------------------------------------------------
# Runner label to GPU mapping
# ---------------------------------------------------------------------------

RUNNER_TO_GPU: dict[str, str] = {
    "max1550": "pvc",
    "b580": "bmg",
}


def resolve_gpus(runner_label: str, history_dir: Path) -> dict[str, Path]:
    """Return a mapping of gpu_name -> history.json path to process.

    If *runner_label* is empty, return all history files found under
    *history_dir*/<gpu>/history.json.
    """
    if runner_label:
        gpu = RUNNER_TO_GPU.get(runner_label)
        if gpu is None:
            print(f"Warning: unknown runner label '{runner_label}', scanning all GPUs", file=sys.stderr)
            return _scan_all(history_dir)
        path = history_dir / gpu / "history.json"
        if not path.exists():
            return {}
        return {gpu: path}
    return _scan_all(history_dir)


def _scan_all(history_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not history_dir.is_dir():
        return result
    for child in sorted(history_dir.iterdir()):
        hf = child / "history.json"
        if child.is_dir() and hf.exists():
            result[child.name] = hf
    return result


# ---------------------------------------------------------------------------
# Key parsing helpers
# ---------------------------------------------------------------------------


def parse_benchmark_name(key: str) -> str:
    """Extract the benchmark name (first segment before '/')."""
    return key.split("/", 1)[0]


def parse_params(key: str) -> str:
    """Extract the params portion (everything after the second '/')."""
    parts = key.split("/", 2)
    return parts[2] if len(parts) > 2 else ""


# ---------------------------------------------------------------------------
# Driver change detection
# ---------------------------------------------------------------------------


def detect_driver_change(
    current_run: dict[str, Any],
    baseline_run: dict[str, Any],
) -> list[dict[str, str]] | None:
    """Compare driver versions between current and most recent baseline run.

    Returns a dict describing the change, or None if no change.
    """
    changes: list[dict[str, str]] = []
    for field_name in ("agama_version", "libigc1_version"):
        cur = current_run.get(field_name, "")
        base = baseline_run.get(field_name, "")
        if cur != base:
            changes.append({"field": field_name, "from": base, "to": cur})
    if not changes:
        return None
    return changes


# ---------------------------------------------------------------------------
# Core regression detection
# ---------------------------------------------------------------------------


@dataclass
class GpuReport:
    """Aggregated report for one GPU."""

    regressions: list[dict[str, Any]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0
    total_checked: int = 0
    driver_change: Any = None


def _collect_ci_runs(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the runs tagged 'ci', in chronological order."""
    return [r for r in history if r.get("tag") == "ci"]


def analyze_gpu(history: list[dict[str, Any]], config: Config) -> GpuReport:
    """Analyze the most recent run against rolling CI baseline."""
    if not history:
        return GpuReport()

    current_run = history[-1]
    is_pr = current_run.get("tag", "").startswith("pr-")

    ci_runs = _collect_ci_runs(history)

    # For PR runs the current run is not necessarily CI-tagged,
    # so the baseline is the full set of ci_runs.
    # For CI runs the baseline excludes the current run itself.
    if is_pr:
        baseline_runs = ci_runs
    else:
        baseline_runs = [r for r in ci_runs if r.get("run_id") != current_run.get("run_id")]

    if not baseline_runs:
        # No CI baseline at all -- nothing to compare.
        all_keys = set(current_run.get("results", {}).keys())
        return GpuReport(skipped=len(all_keys), total_checked=len(all_keys))

    # Driver change: compare against most recent baseline run.
    driver_change = detect_driver_change(current_run, baseline_runs[-1])

    current_results = current_run.get("results", {})
    report = GpuReport(driver_change=driver_change)

    for key, metrics in current_results.items():
        current_tflops = metrics.get("tflops")
        if current_tflops is None:
            continue

        report.total_checked += 1
        benchmark = parse_benchmark_name(key)
        thresh = config.for_benchmark(benchmark)

        # Collect baseline values for this key.
        window = baseline_runs[-thresh.rolling_window:]
        baseline_values = [
            r["results"][key]["tflops"] for r in window if key in r.get("results", {}) and "tflops" in r["results"][key]
        ]

        if len(baseline_values) < thresh.min_history:
            report.skipped += 1
            continue

        # Skip metrics with high coefficient of variation (bimodal/noisy).
        baseline_mean = statistics.mean(baseline_values)
        if baseline_mean > 0:
            cv = statistics.stdev(baseline_values) / baseline_mean
            if cv > thresh.max_cv:
                report.skipped += 1
                continue

        # Dual-median baseline: lock in at recent level if a significant
        # and stable improvement is detected.
        full_median = statistics.median(baseline_values)
        lock_window = max(thresh.min_history, min(8, len(baseline_values)))
        recent_values = baseline_values[-lock_window:]
        recent_median = statistics.median(recent_values)

        # Only lock if recent values are stable (low variability).
        recent_mad = statistics.median([abs(v - recent_median) for v in recent_values])
        is_stable_improvement = (
            recent_median > full_median * (1 + thresh.improvement_lock_pct / 100)
            and recent_mad < recent_median * 0.05  # Less than 5% variability
        )

        if is_stable_improvement:
            baseline_median = recent_median
            baseline_mad = recent_mad if recent_mad > 0 else recent_median * 0.01
        else:
            baseline_median = full_median
            baseline_mad = statistics.median([abs(v - full_median) for v in baseline_values])
        if baseline_mad == 0:
            # Use 1% of baseline as MAD proxy for perfectly stable benchmarks.
            # Keeps z-scores interpretable (e.g., 5% drop -> z ~ -3.4).
            baseline_mad = baseline_median * 0.01 if baseline_median > 0 else 1e-6

        modified_z = 0.6745 * (current_tflops - baseline_median) / baseline_mad

        if baseline_median == 0:
            if current_tflops == 0:
                # Both zero -- nothing to report.
                report.skipped += 1
                continue
            # Non-zero current from zero baseline -- use epsilon for relative calc.
            baseline_median_for_pct = 1e-6
        else:
            baseline_median_for_pct = baseline_median

        relative_drop_pct = (current_tflops - baseline_median) / baseline_median_for_pct * 100

        is_regression = (modified_z < -thresh.z_threshold) and (relative_drop_pct < -thresh.min_drop_pct)
        is_improvement = (modified_z > thresh.z_threshold) and (relative_drop_pct > thresh.min_drop_pct)

        if is_regression:
            report.regressions.append({
                "key": key,
                "benchmark": benchmark,
                "params": parse_params(key),
                "current_tflops": round(current_tflops, 4),
                "baseline_median": round(baseline_median, 4),
                "drop_pct": round(relative_drop_pct, 1),
                "modified_z": round(modified_z, 1),
                "driver_change": driver_change,
            })
        elif is_improvement:
            report.improvements.append({
                "key": key,
                "benchmark": benchmark,
                "params": parse_params(key),
                "current_tflops": round(current_tflops, 4),
                "baseline_median": round(baseline_median, 4),
                "gain_pct": round(relative_drop_pct, 1),
                "modified_z": round(modified_z, 1),
                "driver_change": driver_change,
            })

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report(history_dir: Path, runner_label: str, config: Config) -> dict[str, Any]:
    """Build the full regression report across requested GPUs."""
    gpu_paths = resolve_gpus(runner_label, history_dir)

    # Use the most recent run across all GPUs for top-level metadata.
    latest_run: dict[str, Any] = {}
    gpus_output: dict[str, Any] = {}

    for gpu_name, hpath in gpu_paths.items():
        with open(hpath, encoding="utf-8") as f:
            history: list[dict[str, Any]] = json.load(f)

        if not history:
            continue

        current_run = history[-1]
        if not latest_run or current_run.get("datetime", "") > latest_run.get("datetime", ""):
            latest_run = current_run

        report = analyze_gpu(history, config)
        gpus_output[gpu_name] = {
            "regressions": report.regressions,
            "improvements": report.improvements,
            "skipped": report.skipped,
            "total_checked": report.total_checked,
            "driver_change": report.driver_change,
        }

    return {
        "run_id": latest_run.get("run_id", ""),
        "datetime": latest_run.get("datetime", ""),
        "commit_sha": latest_run.get("commit_sha", ""),
        "gpus": gpus_output,
    }
