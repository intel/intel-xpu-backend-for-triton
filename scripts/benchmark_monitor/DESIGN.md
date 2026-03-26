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

## Detection Algorithm

### Modified Z-Score over mean/stdev

| Approach | Pros | Cons |
|----------|------|------|
| Mean + stdev | Simple | Single outlier skews both |
| **Modified Z-Score (chosen)** | Robust — median + MAD | Assumes unimodal (mitigated by CV filter) |

### Dual gate (statistical + practical)

Both conditions must hold: `modified_z < -3.0` AND `drop_pct < -5.0%`.
Prevents false positives from tiny stable changes and large noisy fluctuations.

### Dual-median baseline with improvement lock-in

Locks baseline at recent median when improvement > 8% and stable (MAD < 5%).
Prevents the rolling median from masking regressions after genuine improvements.

### CV filter

Skips metrics with CV > 15%. Catches bimodal/autotuning patterns where
Modified Z-Score would produce false positives.

## Storage Backends

| Backend | Format | Status | Notes |
|---------|--------|--------|-------|
| `JsonFileBackend` | `<gpu>/history.json` | Stable | Backward compatible with original design |
| `ParquetBackend` | `<gpu>/history.parquet` | Stable | Pandas-native, better for large datasets |
| `DatabaseBackend` | — | Stub | `NotImplementedError`, planned for future |

All backends implement `HistoryBackend` ABC: `load()`, `save()`, `list_gpus()`.
Deduplication (by run_id) and pruning (to 200 entries) happen in the storage layer.

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
