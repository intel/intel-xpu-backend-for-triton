# Design Summary — Benchmark Regression Detection (PR #6442)

Reviewer-facing summary of key design decisions, open questions, and risk areas.
For usage, algorithm details, and troubleshooting, see [README.md](README.md).

## Purpose

Automated CI system that tracks Triton benchmark TFLOPS over time on PVC and BMG,
detects statistically significant regressions using Modified Z-Score, and reports
them as GitHub issues (CI) or PR comments (pull requests). ~3800 LOC, 134 unit
tests, zero external infrastructure.

## Architecture at a Glance

```
triton-benchmarks-{pvc,bmg}.yml          (existing, produces CSV + metadata.json)
        |  workflow_run trigger
        v
triton-benchmark-regressions.yml          (new workflow, write permissions)
        |
        +-- convert:  CSV -> history.json  (on benchmark-data branch)
        +-- detect:   history -> Modified Z-Score analysis -> report.json
        +-- report:   report -> GitHub issue / PR comment
        +-- push:     benchmark-data branch (5-retry backoff)
```

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

### 5. Orphan `benchmark-data` branch for history storage

| Option | Verdict | Notes |
|--------|---------|-------|
| **Orphan branch (chosen)** | Adopted | Zero infra, invisible to devs, built-in audit trail |
| GitHub Artifacts | Rejected | 90-day retention limit |
| Actions Cache | Rejected | 7-day eviction, not designed for persistent data |
| External DB | Rejected | Adds infra/secrets overhead; ~3.5 MB/year doesn't justify it |

Data volume: ~2 commits/day (PVC + BMG), ~3.5 MB/year. Orphan branch has no
shared history with `main`; `git clone`/`pull` never fetches it. JSON files
migrate trivially to external storage if needed later.

### 6. Per-benchmark threshold overrides (YAML)

`thresholds.yaml` provides code-level defaults plus per-benchmark overrides.
Overrides inherit all defaults and only specify parameters that differ. This
avoids hardcoding benchmark-specific knowledge in Python code.

### 7. `gh` CLI wrapper with graceful degradation

All GitHub interactions use a `_run_gh()` wrapper that logs warnings on failure
but never raises. If issue/comment creation fails, the rest of the pipeline
continues. This prevents transient GitHub API issues from failing the CI job.

### 8. Package structure modeled after `triton_utils`

Installable package with `pyproject.toml`, `Config` dataclass, `ActionRunner` ABC.
Dependencies (`pandas`, `pyyaml`) declared in `pyproject.toml`, not hardcoded in
the workflow. Requires Python >= 3.10.

## Open Questions for Reviewers

| # | Topic | Status | Notes |
|---|-------|--------|-------|
| 1 | **History storage** (orphan branch) | Needs team alignment | vlad-penkin flagged this for offline discussion. Author provided justification (see Decision 5 above). |
| 2 | **End-to-end validation** | Pending post-merge | 134 unit tests pass + dry run against bootstrapped PVC data. Real CI validation requires merge. |
| 3 | **`hbm_gbs` metric** | Deferred (TODO in code) | Only `tflops` is collected. Adding `hbm_gbs` is straightforward but deferred to avoid scope creep. |
| 4 | **History cap at 200 entries** | Needs validation | At 1 CI run/day/GPU, 200 entries ≈ 200 days. Sufficient? Or should it be configurable? |
| 5 | **Fork PR security** | By design | Fork PRs skip reporting (no `GITHUB_TOKEN`). Detection still runs; results only in artifacts. |

## Risk Areas

### Concurrent push race
PVC and BMG CI can complete simultaneously, causing push conflicts on
`benchmark-data`. **Mitigation**: 5 retry attempts with `fetch + rebase + push`
and exponential backoff (`sleep(attempt * 2)`). Risk: if >5 concurrent runs
overlap, data is lost for that run (but not corrupted).

### MAD = 0 fallback
For perfectly stable benchmarks (identical TFLOPS across all runs), MAD = 0.
**Mitigation**: uses 1% of baseline median as proxy. Risk: a 1% change on a
perfectly stable benchmark would need `modified_z = 0.6745 * delta / (0.01 * median)`,
so a 5% drop gives z ≈ 3.37, which correctly triggers. Seems reasonable.

### Config drift
`thresholds.yaml` defaults (min_history=8, rolling_window=20, improvement_lock_pct=8.0)
differ from code-level fallback defaults (5, 10, 10.0). If YAML is missing,
behavior changes silently. **Mitigation**: code logs a warning when YAML is absent.
Consider aligning code defaults with YAML defaults.

### Report table truncation
PR comments cap regression/improvement tables at 20 rows. If a driver change
causes >20 regressions, some are silently omitted from the PR comment (but all
appear in the JSON report artifact).

## Test Coverage

**134 tests across 4 files:**

| File | Tests | Covers |
|------|-------|--------|
| `test_detect.py` | ~90 | Core algorithm: Z-score, dual gate, lock-in, CV filter, MAD=0, baseline=0 |
| `test_convert.py` | ~20 | CSV parsing, NaN filtering, platform detection, history I/O |
| `test_report.py` | ~15 | Table formatting, sorting, param display |
| `test_bootstrap.py` | ~9 | GPU detection, report parsing |

**Notable gaps**:
- No integration test for the full `convert -> detect -> report` pipeline
- No test for concurrent push retry logic (shell script, not Python)
- `entry_point.py` argument parsing not directly tested (covered implicitly)

## File Map

| File | Purpose |
|------|---------|
| `.github/workflows/triton-benchmark-regressions.yml` | CI workflow: download artifacts, run pipeline, push history |
| `scripts/benchmark_monitor/pyproject.toml` | Package metadata, dependencies (pandas, pyyaml) |
| `scripts/benchmark_monitor/thresholds.yaml` | Detection thresholds: defaults + per-benchmark overrides |
| `scripts/benchmark_monitor/README.md` | User/developer documentation |
| `scripts/benchmark_monitor/benchmark_monitor/__init__.py` | Package init |
| `scripts/benchmark_monitor/benchmark_monitor/entry_point.py` | CLI dispatcher: argparse + ActionRunner ABC |
| `scripts/benchmark_monitor/benchmark_monitor/convert_results.py` | CSV -> history.json (platform detection, dedup, pruning) |
| `scripts/benchmark_monitor/benchmark_monitor/detect_regressions.py` | Core algorithm: Modified Z-Score, dual gate, lock-in, CV filter |
| `scripts/benchmark_monitor/benchmark_monitor/report_results.py` | GitHub reporting: issues (CI), PR comments, stdout (other) |
| `scripts/benchmark_monitor/benchmark_monitor/bootstrap_history.py` | One-time backfill from past CI runs |
| `scripts/benchmark_monitor/test/__init__.py` | Test package init |
| `scripts/benchmark_monitor/test/test_convert.py` | Tests for convert subcommand |
| `scripts/benchmark_monitor/test/test_detect.py` | Tests for detection algorithm (bulk of test suite) |
| `scripts/benchmark_monitor/test/test_report.py` | Tests for reporting/formatting |
| `scripts/benchmark_monitor/test/test_bootstrap.py` | Tests for bootstrap subcommand |
