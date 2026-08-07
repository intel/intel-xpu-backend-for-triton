"""Layer 4 analysis/detection — pure computation, no I/O.

Implements Modified Z-Score regression and improvement detection against
a rolling CI baseline.  Every function in this module is a pure computation:
no file access, no subprocess calls, no network requests.

Depends only on ``benchmark_monitor.model`` (Layer 0) and the stdlib
``statistics`` module.
"""
# pylint: disable=too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

import statistics

from benchmark_monitor.model import (
    AnalysisResult,
    BenchmarkEntry,
    BenchmarkHistory,
    DetectionConfig,
    MetricResult,
    parse_benchmark_name,
    parse_params,
)

# ---------------------------------------------------------------------------
# Driver change detection
# ---------------------------------------------------------------------------


def detect_driver_change(
    current: BenchmarkEntry,
    baseline: BenchmarkEntry,
) -> list[dict[str, str]] | None:
    """Compare driver versions between *current* and *baseline* entries.

    Delegates to ``BenchmarkEntry.driver_changes_from()``.
    Kept as a module-level function for backward compatibility.
    """
    return current.driver_changes_from(baseline)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_ci_entries(entries: list[BenchmarkEntry]) -> list[BenchmarkEntry]:
    """Return only the entries tagged ``'ci'``, in chronological order."""
    return [e for e in entries if e.tag == "ci"]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze(history: BenchmarkHistory, config: DetectionConfig) -> AnalysisResult:
    """Analyze the most recent entry against a rolling CI baseline.

    The algorithm is a direct port of ``detect_regressions.analyze_gpu``
    adapted to use typed domain objects from ``benchmark_monitor.model``.
    """
    if not history.entries:
        return AnalysisResult()

    current_entry = history.entries[-1]
    is_pr = current_entry.tag.startswith("pr-")

    ci_entries = _collect_ci_entries(history.entries)

    # For PR runs the current run is not necessarily CI-tagged,
    # so the baseline is the full set of ci_entries.
    # For CI runs the baseline excludes the current run itself.
    if is_pr:
        baseline_entries = ci_entries
    else:
        baseline_entries = [e for e in ci_entries if e.run_id != current_entry.run_id]

    if not baseline_entries:
        # No CI baseline at all -- nothing to compare.
        all_keys = set(current_entry.results.keys())
        return AnalysisResult(skipped=len(all_keys), total_checked=len(all_keys))

    # Driver change: compare against most recent baseline entry.
    driver_change = detect_driver_change(current_entry, baseline_entries[-1])

    current_results = current_entry.results
    result = AnalysisResult(driver_change=driver_change)

    for key, metrics in current_results.items():
        current_tflops = metrics.get("tflops")
        if current_tflops is None:
            continue

        result.total_checked += 1
        benchmark = parse_benchmark_name(key)
        thresh = config.for_benchmark(benchmark)

        # Collect baseline values for this key.
        window = baseline_entries[-thresh.rolling_window:]
        baseline_values = [e.results[key]["tflops"] for e in window if key in e.results and "tflops" in e.results[key]]

        if len(baseline_values) < thresh.min_history:
            result.skipped += 1
            continue

        # Skip metrics with high coefficient of variation (bimodal/noisy).
        baseline_mean = statistics.mean(baseline_values)
        if baseline_mean > 0:
            cv = statistics.stdev(baseline_values) / baseline_mean
            if cv > thresh.max_cv:
                result.skipped += 1
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
                result.skipped += 1
                continue
            # Non-zero current from zero baseline -- use epsilon for relative calc.
            baseline_median_for_pct = 1e-6
        else:
            baseline_median_for_pct = baseline_median

        relative_drop_pct = (current_tflops - baseline_median) / baseline_median_for_pct * 100

        is_regression = (modified_z < -thresh.z_threshold) and (relative_drop_pct < -thresh.min_drop_pct)
        is_improvement = (modified_z > thresh.z_threshold) and (relative_drop_pct > thresh.min_drop_pct)

        if is_regression:
            result.regressions.append(
                MetricResult(
                    key=key,
                    benchmark=benchmark,
                    params=parse_params(key),
                    current_tflops=round(current_tflops, 4),
                    baseline_median=round(baseline_median, 4),
                    change_pct=round(relative_drop_pct, 1),
                    modified_z=round(modified_z, 1),
                    driver_change=driver_change,
                ))
        elif is_improvement:
            result.improvements.append(
                MetricResult(
                    key=key,
                    benchmark=benchmark,
                    params=parse_params(key),
                    current_tflops=round(current_tflops, 4),
                    baseline_median=round(baseline_median, 4),
                    change_pct=round(relative_drop_pct, 1),
                    modified_z=round(modified_z, 1),
                    driver_change=driver_change,
                ))

    return result
