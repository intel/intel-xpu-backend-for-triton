"""Compare every long-format vLLM benchmark report from the paired GRF jobs."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median

KEY_FIELDS = ("report", "benchmark_group", "benchmark", "compiler", "value_name", "params")
ENV_FIELDS = ("gpu_device", "libigc1_version", "level_zero_version", "torch_version", "benchmarking_method")
EXPECTED_REPORTS = {
    f"{family}{dtype}{patch}-report.csv"
    for family, dtypes in (("unified-attention-performance", ("", "-fp8")), ("moe-gemm-performance", ("", "-fp8")),
                           ("fused-moe-gemm-performance", ("", )))
    for dtype in dtypes
    for patch in ("", "-td")
}


def load_reports(directory):
    samples = defaultdict(list)
    environment = defaultdict(set)
    files = sorted(directory.rglob("*-report.csv"))
    for path in files:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("benchmark_group") != "vllm":
                    continue
                row["params"] = json.dumps(json.loads(row["params"]), sort_keys=True, separators=(",", ":"))
                row["report"] = path.name
                key = tuple(row[field] for field in KEY_FIELDS)
                try:
                    value = float(row["value"])
                except (ValueError, TypeError):
                    value = math.nan
                samples[key].append(value)
                for field in ENV_FIELDS:
                    if row.get(field):
                        environment[field].add(row[field])
    values = {
        key: median(items) if all(math.isfinite(value) and value > 0
                                  for value in items) else math.nan
        for key, items in samples.items()
    }
    return values, environment, [path.name for path in files]


def compare_values(baseline, forced, threshold):
    rows = []
    for key in sorted(baseline.keys() | forced.keys()):
        row = dict(zip(KEY_FIELDS, key))
        before, after = baseline.get(key), forced.get(key)
        row.update(default_value=before, forced_256_value=after, speedup=None, improvement_percent=None)
        if before is None or after is None:
            row["status"] = "missing_default" if before is None else "missing_256"
        elif not (math.isfinite(before) and math.isfinite(after)):
            row["status"] = "invalid_measurement"
        else:
            metric = row["value_name"]
            if metric in ("time_us", "time_ms", "ns_per_arg"):
                ratio = before / after
            elif metric in ("tflops", "gbps"):
                ratio = after / before
            else:
                row["status"] = "unclassified_metric"
                rows.append(row)
                continue
            improvement = (ratio - 1) * 100
            row.update(speedup=ratio, improvement_percent=improvement)
            row["status"] = "improved" if improvement > threshold else (
                "regressed" if improvement < -threshold else "within_threshold")
        rows.append(row)
    return rows


def compare(baseline_dir, forced_dir, output_dir, threshold):
    baseline, baseline_env, baseline_files = load_reports(baseline_dir)
    forced, forced_env, forced_files = load_reports(forced_dir)
    rows = compare_values(baseline, forced, threshold)
    env_differences = {
        field: {"default": sorted(baseline_env[field]), "256": sorted(forced_env[field])}
        for field in ENV_FIELDS
        if baseline_env[field] != forced_env[field]
    }
    missing_files = {
        "missing_default": sorted((EXPECTED_REPORTS | set(forced_files)) - set(baseline_files)),
        "missing_256": sorted((EXPECTED_REPORTS | set(baseline_files)) - set(forced_files)),
    }
    coverage_complete = bool(baseline and forced) and not any(missing_files.values()) and not any(
        row["status"] in ("missing_default", "missing_256", "invalid_measurement") for row in rows)
    summary = {
        "main_base": "7b8425bea64754ca5ba1386ffaeacea598bf9c3c",
        "threshold_percent": threshold,
        "metric_rows": len(rows),
        "counts": dict(Counter(row["status"] for row in rows)),
        "default_report_files": baseline_files,
        "forced_256_report_files": forced_files,
        "missing_report_files": missing_files,
        "environment_differences": env_differences,
        "coverage_complete": coverage_complete,
        "environment_matches": not env_differences,
    }
    write_outputs(output_dir, rows, summary)
    return summary


def write_outputs(output_dir, rows, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[*KEY_FIELDS, "default_value", "forced_256_value", "speedup", "improvement_percent", "status"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    text = [
        "# vLLM: default GRF versus forced 256",
        "",
        f"Both jobs use main base `{summary['main_base']}` and the complete default vLLM suite.",
        "Counts below are metric rows; each workload can report latency and throughput separately.",
        f"Improvement is `(speedup - 1) * 100`; positive values favor forced 256. "
        f"Threshold: {summary['threshold_percent']:g}%.",
        "",
        f"Report coverage complete: **{summary['coverage_complete']}**. "
        f"Recorded environments match: **{summary['environment_matches']}**.",
        "",
        "| Report | Provider | Metric | Improved | Regressed | Within threshold | Missing/invalid |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[(row["report"], row["compiler"], row["value_name"])][row["status"]] += 1
    for (report, compiler, metric), counts in sorted(grouped.items()):
        missing = sum(counts[name] for name in ("missing_default", "missing_256", "invalid_measurement"))
        text.append(f"| {report} | {compiler} | {metric} | {counts['improved']} | {counts['regressed']} | "
                    f"{counts['within_threshold']} | {missing} |")
    if summary["environment_differences"] or any(summary["missing_report_files"].values()):
        text.extend([
            "", "```json",
            json.dumps({key: summary[key]
                        for key in ("environment_differences", "missing_report_files")}, indent=2), "```"
        ])
    text.extend([
        "", "The comparison artifact contains every row and its workload parameters in `comparison.csv`.",
        "Failed benchmark steps and workloads missing from both reports must also be checked in the job logs."
    ])
    (output_dir / "summary.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--forced", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()
    result = compare(args.baseline, args.forced, args.output, args.threshold)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["coverage_complete"] and result["environment_matches"] else 1)
