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

Per-benchmark overrides keyed by benchmark name. Each override inherits all
defaults and only specifies parameters that differ.

## Detection Algorithm

### Modified Z-Score

```
modified_z = 0.6745 * (current - baseline_median) / baseline_MAD
```

Robust to outliers. MAD=0 fallback uses 1% of baseline median.

### Dual gate

A metric is flagged only when **both** conditions hold:
1. `modified_z < -z_threshold` (statistically significant)
2. `drop_pct < -min_drop_pct` (practically significant)

### Dual-median baseline

Locks baseline at recent median when a stable improvement is detected,
preventing the rolling median from masking subsequent regressions.

### CV filter

Skips metrics with coefficient of variation > `max_cv` (bimodal/autotuning).

## Running Tests

```bash
cd scripts/benchmark_monitor && uv run pytest test/ -v
```

158 tests across 6 per-layer test files.

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
