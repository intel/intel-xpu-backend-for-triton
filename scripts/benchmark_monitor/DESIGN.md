# Design Summary — Benchmark Regression Detection

Reviewer-facing summary of architecture decisions and design rationale.
For usage and CLI reference, see [README.md](README.md).

## Layered Architecture

The package is organized into strict layers with downward-only dependencies.
No circular imports. Each layer depends only on Layer 0 (domain model).

```
entry_point.py → imports all layers, wires them together
    │
    ├── formatting.py  → imports model.py only     (Layer 5: pure strings)
    ├── analysis.py    → imports model.py only     (Layer 4: pure computation)
    ├── ingest.py      → imports model.py only     (Layer 3: DataFrame → domain)
    ├── source.py      → imports model.py only     (Layer 2a: data acquisition)
    ├── storage.py     → imports model.py only     (Layer 2b: persistence)
    └── model.py       → imports nothing           (Layer 0: dataclasses)
```

### Layer Responsibilities

| Layer | Module | Responsibility | I/O |
|-------|--------|---------------|-----|
| 0 | `model.py` | Domain types, enums, constants, parsing helpers | None |
| 2a | `source.py` | `ReportSource` ABC → `LocalCSVSource`, `GHArtifactSource` | Read CSVs, download artifacts |
| 2b | `storage.py` | `HistoryBackend` ABC → `JsonFileBackend`, `ParquetBackend`, `DatabaseBackend` (stub) | Read/write history files |
| 3 | `ingest.py` | `parse_reports()`: DataFrame → `BenchmarkEntry` per GPU | None |
| 4 | `analysis.py` | `analyze()`: `BenchmarkHistory` → `AnalysisResult` (Modified Z-Score) | None |
| 5 | `formatting.py` | Tables, summaries, issue bodies, PR comments → strings | None |

### Design Principles

1. **Separation of concerns** — analysis never imports storage; formatting never imports analysis
2. **Composable output** — package produces strings, never posts to GitHub directly
3. **Storage abstraction** — backends are interchangeable (JSON ↔ Parquet) with identical results
4. **Dual-use** — every subcommand works locally without CI infrastructure

## Key Design Decisions

### 1. Modified Z-Score over mean/stdev

| Approach | Pros | Cons |
|----------|------|------|
| **Mean + stdev** | Simple, well-known | Single outlier skews both mean and stdev |
| **Modified Z-Score (chosen)** | Robust — uses median + MAD | Assumes unimodal data (mitigated by CV filter) |

The 0.6745 normalization constant makes scores comparable to standard Z-scores.
A threshold of 3.0 corresponds roughly to 99.7% confidence under normality.

### 2. Dual gate (statistical + practical)

Both conditions must hold to flag a regression:
- `modified_z < -3.0` (statistically significant)
- `drop_pct < -5.0%` (practically significant)

**Rationale**: prevents two classes of false positives:
- Tiny absolute changes on extremely stable benchmarks (stat-significant but irrelevant)
- Large one-off fluctuations on noisy benchmarks (large drop but expected)

### 3. Dual-median baseline with improvement lock-in

After a genuine improvement, the rolling median lags behind the new level, masking
subsequent regressions. The detector locks baseline at the recent median when:
- Recent median > full median by `improvement_lock_pct` (default 8%)
- Recent values are stable (MAD < 5% of recent median)

**Alternative considered**: simply shrinking the rolling window — rejected because
it reduces baseline robustness for all metrics, not just improved ones.

### 4. Coefficient of variation (CV) filter

Skips metrics with CV > 15% (bimodal/autotuning). Uses stdev-based CV instead of
MAD-based because stdev is more sensitive to bimodal spread — exactly the pattern
we want to detect.

### 5. Per-benchmark threshold overrides (YAML)

`thresholds.yaml` provides code-level defaults plus per-benchmark overrides.
Overrides inherit all defaults and only specify parameters that differ. This
avoids hardcoding benchmark-specific knowledge in Python code.

### 6. Composable output over direct posting

The package produces formatted strings (`--format` flag) and never calls
`gh issue create` or `gh pr comment` directly. CI workflows compose the output
with `gh` CLI calls. This keeps the package locally useful and decoupled from
GitHub's API surface.

### 7. Storage abstraction

The `HistoryBackend` ABC decouples the detection algorithm from storage format.
JSON and Parquet backends produce identical results. Database backend is stubbed
for future use. This enables storage migration without touching analysis code.

## Storage Backends

| Backend | Format | Status | Notes |
|---------|--------|--------|-------|
| `JsonFileBackend` | `<gpu>/history.json` | Stable | Backward compatible with original design |
| `ParquetBackend` | `<gpu>/history.parquet` | Stable | Pandas-native, better for large datasets |
| `DatabaseBackend` | — | Stub | `NotImplementedError`, planned for future |

All backends implement `HistoryBackend` ABC: `load()`, `save()`, `list_gpus()`.
Deduplication (by run_id) and pruning (to 200 entries) happen in the storage layer.

## Risk Areas

### MAD = 0 fallback
For perfectly stable benchmarks (identical TFLOPS across all runs), MAD = 0.
**Mitigation**: uses 1% of baseline median as proxy. Risk: a 1% change on a
perfectly stable benchmark would need `modified_z = 0.6745 * delta / (0.01 * median)`,
so a 5% drop gives z ≈ 3.37, which correctly triggers. Seems reasonable.

### Config drift (resolved)
The original code had `thresholds.yaml` defaults (min_history=8, rolling_window=20,
improvement_lock_pct=8.0) that differed from code-level fallback defaults (5, 10, 10.0).
**Resolution**: code defaults now match YAML defaults exactly.

### Report table truncation
PR comments cap regression/improvement tables at 20 rows. If a driver change
causes >20 regressions, some are omitted from the PR comment (but all
appear in the JSON report).

### History cap at 200 entries
At 1 CI run/day/GPU, 200 entries ≈ 200 days. Pruning happens in the storage
layer during save. Configurable via `MAX_HISTORY_ENTRIES` constant in `storage.py`.

## Test Coverage

**158 tests across 6 per-layer files:**

| File | Tests | Covers |
|------|-------|--------|
| `test_model.py` | ~30 | Platform detection, key parsing, config defaults, dataclass construction |
| `test_analysis.py` | ~46 | Full algorithm: Z-score, dual gate, lock-in, CV filter, edge cases |
| `test_formatting.py` | ~27 | Tables, sorting, issue bodies, PR comments, summaries |
| `test_storage.py` | ~20 | JSON/Parquet round-trips, dedup, pruning, DB stub |
| `test_ingest.py` | ~14 | CSV parsing, platform grouping, NaN filtering |
| `test_source.py` | ~12 | LocalCSV source, GHArtifact source (mocked) |

E2E validated against 30 real CI runs (PVC + BMG) with both JSON and Parquet backends
producing identical detection results.

## File Map

| File | Purpose |
|------|---------|
| `benchmark_monitor/model.py` | Layer 0: domain dataclasses, enums, constants |
| `benchmark_monitor/source.py` | Layer 2a: ReportSource ABC + LocalCSV/GHArtifact |
| `benchmark_monitor/storage.py` | Layer 2b: HistoryBackend ABC + Json/Parquet/DB |
| `benchmark_monitor/ingest.py` | Layer 3: DataFrame → BenchmarkEntry |
| `benchmark_monitor/analysis.py` | Layer 4: Modified Z-Score detection |
| `benchmark_monitor/formatting.py` | Layer 5: markdown tables, issue bodies, summaries |
| `benchmark_monitor/entry_point.py` | CLI: argparse + wiring |
| `thresholds.yaml` | Detection thresholds with per-benchmark overrides |
| `pyproject.toml` | Package metadata and dependencies |
