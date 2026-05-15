# Benchmark Regression Detection

Performance regression detection for the Intel XPU Backend for Triton.
Tracks benchmark TFLOPS over time per GPU platform and flags statistically
significant regressions using a Modified Z-Score algorithm. Works both locally
and in CI — the package produces structured reports and formatted output,
leaving posting decisions to the caller.

## Installation

```bash
pip install ./scripts/benchmark_monitor
```

## Architecture

The package follows a layered architecture with strict downward-only dependencies:

```
┌─────────────────────────────────────────────────────┐
│  CLI / Entry Point (entry_point.py)                 │  Wiring layer
├─────────────────────────────────────────────────────┤
│  Layer 5: Formatting (formatting.py)                │  Pure string generation
├─────────────────────────────────────────────────────┤
│  Layer 4: Analysis (analysis.py)                    │  Pure computation
├─────────────────────────────────────────────────────┤
│  Layer 3: Ingestion (ingest.py)                     │  DataFrame → domain objects
├──────────────────────┬──────────────────────────────┤
│  Layer 2a: Source    │  Layer 2b: Storage            │  I/O boundaries
│  (source.py)         │  (storage.py)                │
├──────────────────────┴──────────────────────────────┤
│  Layer 0: Domain Model (model.py)                   │  Pure dataclasses
└─────────────────────────────────────────────────────┘
```

Each layer depends only on Layer 0 (model). The entry point wires them together.

## CLI Subcommands

### benchmark-monitor convert

Reads `*-report.csv` files, detects GPU platform, and appends entries to history.

```bash
benchmark-monitor convert \
    --reports-dir ./reports \
    --history-dir ./benchmark-data \
    --tag ci \
    --run-id 12345678 \
    --commit-sha abc123 \
    --backend json        # or: parquet
```

### benchmark-monitor detect

Analyzes the most recent run against a rolling CI baseline. Writes a JSON report
and exits non-zero if regressions are found.

```bash
benchmark-monitor detect \
    --history-dir ./benchmark-data \
    --output ./report.json \
    --config scripts/benchmark_monitor/thresholds.yaml \
    --backend json \
    --format json         # or: text, markdown, issue-title, issue-body, pr-comment
```

**Output formats** (`--format`):

| Format | Stdout output | Use case |
|--------|--------------|----------|
| `json` (default) | Summary to stderr only | CI pipelines reading report.json |
| `text` | Plain text summary | Local inspection |
| `markdown` | Markdown regression/improvement tables | Documentation, notebooks |
| `issue-title` | GitHub issue title string | `gh issue create --title "$(...)"` |
| `issue-body` | Full markdown issue body | `gh issue create --body "$(...)"` |
| `pr-comment` | PR comment body with marker | `gh pr comment --body "$(...)"` |

Use `--gpu pvc` (or `bmg`) with `issue-title` and `issue-body` to target a specific GPU.

The JSON report file (`--output`) is always written regardless of `--format`.

### benchmark-monitor bootstrap

Seeds history from past GitHub Actions runs:

```bash
benchmark-monitor bootstrap \
    --history-dir ./benchmark-data \
    --max-runs 30 \
    --workflow triton-benchmarks.yml \
    --repo intel/intel-xpu-backend-for-triton \
    --backend json \
    --dry-run             # list runs without downloading
```

## Storage Backends

The `--backend` flag selects the history storage format:

| Backend | File | Status |
|---------|------|--------|
| `json` (default) | `<gpu>/history.json` | Stable, backward compatible |
| `parquet` | `<gpu>/history.parquet` | Stable, pandas-native |
| `db` | — | Planned for future release |

Both `json` and `parquet` backends produce identical detection results.

## Configuration

All detection thresholds are in `thresholds.yaml`. Code defaults are aligned
with the YAML defaults (no silent drift).

### `defaults` section

| Parameter              | Default | Description                                                      |
|------------------------|---------|------------------------------------------------------------------|
| `min_history`          | 8       | Minimum CI runs required before detection activates              |
| `rolling_window`       | 20      | Number of recent CI runs used for baseline computation           |
| `z_threshold`          | 3.0     | Modified Z-Score threshold (roughly 99.7% confidence)            |
| `min_drop_pct`         | 5.0     | Minimum relative performance drop (%) to flag as a regression    |
| `improvement_lock_pct` | 8.0     | Lock baseline at recent level if improvement exceeds this %      |
| `max_cv`               | 0.15    | Skip metrics with coefficient of variation > 15% (bimodal/noisy) |

### `overrides` section

Per-benchmark overrides are keyed by the benchmark name (the first segment of
the metric key, matching the `--benchmark` name used in `build_report.py`).
Each override inherits all defaults and only needs to specify the parameters
that differ.

Current overrides:

| Benchmark                | Override        | Reason                                       |
|--------------------------|-----------------|----------------------------------------------|
| `softmax`                | min_drop_pct: 3 | Very stable benchmark; tighter threshold     |
| `flash-attn`             | min_drop_pct: 8 | Noisy due to kernel complexity               |
| `flash-attn-bwd`         | min_drop_pct: 8 | Noisy due to kernel complexity               |
| `flex-attn-causal`       | min_drop_pct: 8 | Noisy due to mask complexity and autotuning  |
| `flex-attn-causal-batch4`  | min_drop_pct: 8 | Same as above                              |
| `flex-attn-causal-batch16` | min_drop_pct: 8 | Same as above                              |
| `flex-attn-causal-bwd`  | min_drop_pct: 8 | Same as above                                |
| `flex-attn-masks`        | min_drop_pct: 8 | Same as above                                |

## Detection Algorithm

### Modified Z-Score

The detector uses the Modified Z-Score, which is robust to outliers:

```
modified_z = 0.6745 * (current - baseline_median) / baseline_MAD
```

where `baseline_median` may be the full-window or recent-window median (see
dual-median baseline below), and MAD is the Median Absolute Deviation. The constant
0.6745 normalizes the score so that it is comparable to a standard Z-score for
normally distributed data. If MAD is zero (perfectly stable benchmark), a proxy
of 1% of the median is used.

### Dual gate

A metric is flagged as a regression only when **both** conditions are met:

1. `modified_z < -z_threshold` (statistically significant deviation)
2. `relative_drop_pct < -min_drop_pct` (practically significant drop)

This avoids false positives from tiny absolute changes that happen to be
statistically significant, and from large but statistically expected fluctuations
in noisy benchmarks. The same logic (with reversed signs) applies to improvements.

### Dual-median baseline

When a benchmark has a genuine recent improvement, the rolling median lags behind
the new performance level, causing the improved values to mask real regressions.
To counter this, the detector checks whether the recent window of values is both:

- Significantly higher than the full-window median (by `improvement_lock_pct`)
- Stable (MAD < 5% of the recent median)

When both conditions hold, the baseline locks in at the recent median, making
regressions from the new level detectable immediately.

### Coefficient of variation filter

Before analyzing a metric, the detector computes the coefficient of variation
(CV = stdev / mean) of the baseline values. If CV exceeds `max_cv` (default
0.15), the metric is skipped as too noisy for reliable analysis. This catches
bimodal distributions (e.g. autotuning alternating between two performance
levels) where the Modified Z-Score — designed for unimodal data — would produce
false positives. Standard deviation is used instead of MAD because it is more
sensitive to bimodal spread.

### CI vs PR runs

CI runs (tagged `ci`) form the baseline. When analyzing a CI run, it is excluded
from its own baseline. PR runs (tagged `pr-<number>`) are compared against the
full CI baseline without contributing to it.

## Adding a New Benchmark

1. Add a `build_report.py` call in `triton-benchmarks.yml` with
   `--benchmark <name>` to produce the CSV report.
2. Optionally add an override entry in `thresholds.yaml` if the benchmark is
   noisy or unusually stable.
3. No other changes are needed. The benchmark name (first segment of the metric
   key, before the first `/`) is used automatically for override matching.

## Running Tests

```bash
cd scripts/benchmark_monitor && uv run pytest test/ -v
```

158 tests across 6 per-layer test files.

## Troubleshooting

**Noisy benchmarks producing false positives.**
Add or increase `min_drop_pct` for the benchmark in the `overrides` section of
`thresholds.yaml`. Values of 8-10% work well for attention-family benchmarks.

**Bimodal benchmarks (e.g. autotuning alternates between two levels).**
Metrics with coefficient of variation > `max_cv` (default 15%) are automatically
skipped. If a benchmark is known to be bimodal but its CV is borderline, lower
`max_cv` for that benchmark in the `overrides` section. If it is stable but
being skipped, raise `max_cv`.

**Investigating a flagged regression.**
Inspect the `regression-report.json` artifact from the CI run. Each entry
includes `modified_z`, `drop_pct`, `baseline_median`, and `current_tflops`.
Compare these against the threshold config to confirm or dismiss the flag.

**History pruning.**
Each GPU history file is capped at 200 entries. Older entries are pruned
automatically by the storage backend when saving.

**Driver change notifications.**
When a regression coincides with a driver version change (agama or libigc1),
the report includes a notice. This helps distinguish compiler regressions from
driver-caused changes.

## E2E Validation

```bash
# Bootstrap from real CI runs
benchmark-monitor bootstrap --history-dir /tmp/e2e --max-runs 30

# Run detection
benchmark-monitor detect --history-dir /tmp/e2e --output /tmp/e2e/report.json \
    --config scripts/benchmark_monitor/thresholds.yaml

# Inspect results
benchmark-monitor detect ... --format text
benchmark-monitor detect ... --format markdown
benchmark-monitor detect ... --format issue-title --gpu pvc
```
