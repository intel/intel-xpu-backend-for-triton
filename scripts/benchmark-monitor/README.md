# Benchmark Regression Detection

Automated performance regression detection for the Intel XPU Backend for Triton.
The system runs inside GitHub Actions CI, tracks benchmark TFLOPS over time per
GPU platform, and flags statistically significant regressions using a Modified
Z-Score algorithm. Results are posted as GitHub issues (CI runs) or PR comments
(pull request runs).

## Architecture

Benchmark execution and regression detection are split across two workflows:

1. **`triton-benchmarks.yml`** (reusable, called by `triton-benchmarks-pvc.yml`
   and `triton-benchmarks-bmg.yml`): runs benchmarks, produces CSV reports, and
   uploads a `benchmark-reports` artifact including a `metadata.json` file.
2. **`triton-benchmark-regressions.yml`** (triggered by `workflow_run` on
   completion of the above): downloads the artifact, runs the detection scripts,
   and posts results. This workflow declares its own write permissions, so the
   caller workflows only need `read-all`.

## Data Flow

```
triton-benchmarks-{pvc,bmg}.yml (caller, permissions: read-all)
  |
  v
triton-benchmarks.yml (reusable workflow)
  |  build_report.py (in benchmarks/triton_kernels_benchmark/) produces
  |  *-report.csv files + metadata.json
  |  uploads benchmark-reports artifact
  |
  v  (workflow_run trigger on completion)
triton-benchmark-regressions.yml (permissions: contents+issues+PRs write)
  |  downloads benchmark-reports artifact via gh CLI
  |
  v
convert_results.py
  - reads *-report.csv files, filters to compiler=triton
  - detects GPU platform (PVC / BMG) from gpu_device column
  - appends a timestamped entry to <gpu>/history.json on the benchmark-data branch
  |
  v
detect_regressions.py
  - loads <gpu>/history.json
  - builds a rolling baseline from CI-tagged runs
  - compares the most recent run against the baseline using Modified Z-Score
  - writes regression-report.json
  |
  v
report_results.py
  - CI runs:  creates or updates a GitHub issue labeled "perf-regression"
  - PR runs:  posts or updates a PR comment with a benchmark results table
  - Other:    prints a summary to stdout
```

## Scripts

### convert_results.py

Parses `*-report.csv` files and appends the current run to the per-GPU
`history.json`. Deduplicates by run ID and prunes history to 200 entries.

Key arguments: `--reports-dir`, `--history-dir`, `--tag`, `--run-id`, `--commit-sha`

### detect_regressions.py

Analyzes the most recent run against the rolling CI baseline and writes a
JSON report. Exits non-zero if any regressions are detected (for CI gating).

Key arguments: `--history-dir`, `--runner-label` (e.g. `max1550`, `b580`),
`--output`, `--config` (path to `thresholds.yaml`)

Runner labels map to GPU platform directories:

| Runner Label | GPU Platform |
|--------------|--------------|
| `max1550`    | `pvc`        |
| `b580`       | `bmg`        |

If the label is empty or unknown, all GPUs under `history-dir/*/history.json`
are analyzed.

### report_results.py

Reads the regression report JSON and posts results to GitHub. Behavior
depends on the `--tag` value: `ci` creates issues, `pr-*` posts PR comments.

Key arguments: `--report`, `--tag`, `--pr-number`, `--run-url`, `--repo`

### bootstrap_history.py

One-time setup script that downloads benchmark artifacts from past GitHub
Actions runs and populates the initial `history.json` files.

Key arguments: `--history-dir`, `--max-runs`, `--workflow`, `--actor`, `--dry-run`

## Configuration

All detection thresholds are in `thresholds.yaml`. These override the code-level
fallback defaults (min_history=5, rolling_window=10, improvement_lock_pct=10.0).

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
cd scripts/benchmark-monitor && python -m pytest tests/ -v
```

## Troubleshooting

**Missing benchmark-data branch.**
The history files live on a separate `benchmark-data` branch. On the first CI
run, the workflow creates the branch if it does not exist. `convert_results.py`
then creates the GPU subdirectories (`pvc/`, `bmg/`) and `history.json` files
when appending data. For initial seeding from past runs, use `bootstrap_history.py`.

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
automatically by `convert_results.py` when appending new data.

**Concurrent CI runs causing push conflicts.**
The `benchmark-data` branch push uses 5 retry attempts with exponential backoff.
Each retry fetches and rebases before pushing. This handles the case where PVC
and BMG CI runs complete simultaneously.

**PR from fork does not get benchmark comment.**
For security, PRs from external forks skip the reporting step. The regression
detection still runs, but results are only available in the workflow artifacts.

**Driver change notifications.**
When a regression coincides with a driver version change (agama or libigc1),
the report and GitHub issue include a notice. This helps distinguish
compiler regressions from driver-caused changes.
