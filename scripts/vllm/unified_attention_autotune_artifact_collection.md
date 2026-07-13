# Unified Attention Autotune Artifact Collection

## Overview

The unified-attention comparison runner now records enough information to explain why two branches selected different Triton autotune configurations. Previously, the decision CSV mostly described the winning configuration, which made it difficult to distinguish a genuine tuning difference from a changed candidate set, pruning decision, noisy measurement, cache hit, or mismatched run.

The collection changes are intentionally observational. They do not add autotune key dimensions, change the configured candidate list, or alter the unified-attention kernel itself. The only tuning-related behavior added by the collector is bounded remeasurement of high-variance timings.

## Files involved

- `run_unified_attention_3d_solution_comparison.sh` orchestrates the branch runs, assigns run identity, applies the appropriate collector patch, and summarizes the output.
- `unified_attention_benchmark_autotune_artifacts.patch` is the complete current collector patch for a source tree with no earlier collector installed.
- `unified_attention_benchmark_autotune_artifacts_v1_to_v2.patch` upgrades the older collector already present on `ua-combined-autotune-ARTIFACT` to the current schema and behavior.
- `run_benchmark_patched_only.patch` remains the small benchmark-runner patch used by the comparison workflow.

Keeping both a complete patch and an upgrade patch lets the runner compare `main` with the artifact branch without requiring both branches to start from the same instrumentation state.

## Collection workflow

For each comparison, the runner performs the following sequence:

1. Resolve all requested refs to exact commit hashes.
2. Create one comparison run UUID and UTC timestamp shared by all sides.
3. Create a separate run UUID for each branch/dtype execution.
4. Prepare isolated worktrees and a corresponding vLLM source tree.
5. Detect the collector version in the target source:

   - apply the complete patch when no collector is present;
   - apply the v1-to-v2 patch when the older artifact collector is present;
   - leave an already-current collector unchanged.
6. Export the provenance values and collector settings to the benchmark process.
7. Collect decision, candidate, benchmark, and log artifacts under the branch label.
8. Revert temporary patching during cleanup.

The summary printed at the end includes the comparison run UUID, timestamp, and resolved commits so that directories from different invocations cannot be accidentally treated as one experiment.

## Output layout

A run produces artifacts in the configured output root, grouped by branch label and data type. The important autotune files are:

```text
<output-root>/
  <branch-label>/
    <dtype>/
      ... decision CSV ...
      ... candidate CSV ...
      ... benchmark and log artifacts ...
```

The files are named `unified-attention-autotune-decisions.csv` and `unified-attention-autotune-candidates.csv`. Every row also carries explicit provenance. Consumers should use `run_uuid` to associate decision and candidate rows within one branch/dtype execution. To compare branches, match rows by `comparison_run_uuid`, data type, benchmark variant, workload shape, and autotune key rather than relying on directory names alone.

## Decision CSV

The decision CSV contains one row per observed autotune decision. It records:

- branch label and resolved commit;
- comparison run UUID, per-branch/dtype run UUID, and UTC timestamp;
- workload dimensions and relevant shape fields;
- the autotune key;
- the selected and effective configurations;
- whether `selection_source` was `autotuner`, `manual_cache`, or `fixed`, plus the observed autotune cache size;
- the outer benchmark median, mean, coefficient of variation, and retry metadata;
- the candidate winner-to-runner-up margin when fresh candidate timings are available;
- the candidate configurations before and after pruning, their counts, and any pruning error.

Fixed configurations are emitted as explicit one-candidate decisions. This is important for `main`, where a fixed effective configuration previously looked like missing autotune data.

Cache-derived decisions remain useful for proving which configuration executed, but they may not have fresh candidate timing rows. The candidate CSV marks listener-reported cache reuse with `cache_hit`. When branch comparison requires timing evidence, use a clean autotune cache or a distinct cache location.

## Candidate CSV

The candidate CSV is the detailed evidence behind a decision. It emits one row for every candidate considered by the autotuner, including candidates that never reached benchmarking.

Each row contains:

- the same provenance and workload identity as the decision row;
- candidate configuration parameters;
- candidate status;
- timing quantiles, mean, standard deviation, and coefficient of variation;
- rank among successfully timed candidates;
- the selected-to-runner-up timing margin for the candidate set.

The `pruning_status` values are `benchmarked`, `selected_without_benchmark`, `pruned`, and `fixed`. The file records the actual before/after candidate sets, but it does not currently classify each `pruned` row by an individual pruning reason.

### Actual pruning decisions

Pruning is captured by wrapping the real `prune_configs` call made by the kernel autotuner rather than reconstructing its output later. The decision row also records the before/after configuration lists and a `pruning_error` if the diagnostic replay fails.

The runtime diagnostic also includes the unified-attention tile controls, including `USE_LARGE_HEAD_TILE` and `USE_TD`. Those fields were previously absent from the diagnostic path, which made some apparently missing candidates impossible to explain.

### Timing and noise handling

For a successfully benchmarked candidate, the collector requests all raw timing samples and uses them to derive the values written to the CSV. The CSV stores p20, median, p80, mean, standard deviation, and CV rather than the full raw sample array. The mean and sample standard deviation exclude the lowest and highest samples when more than two samples are available. It calculates:

```text
CV = sample_standard_deviation / sample_mean
```

When the CV exceeds `UA_AUTOTUNE_CV_THRESHOLD`, the candidate is remeasured up to `UA_AUTOTUNE_MAX_CV_RETRIES` times. The measurement attempt with the lowest CV is retained. The default settings are:

```text
UA_AUTOTUNE_CV_THRESHOLD=0.02
UA_AUTOTUNE_MAX_CV_RETRIES=2
```

The same bounded retry policy is used for the outer benchmark measurement. This reduces branch differences caused by transient variance while preserving evidence of the retained measurement in the CSV.

Ranks are assigned only among successfully timed candidates. The winner has rank 1, and the selected-to-runner-up percentage is calculated as:

```text
selected_runner_up_margin_pct = (runner_up_time / winner_time - 1) * 100
```

The margin is copied to the decision and candidate rows for that candidate set. A small margin means the selected configuration is sensitive to measurement noise and should not be treated as a strong branch-level difference without repetition.

## Provenance and comparability

The following identity fields are injected by the runner and copied into every relevant artifact row:

- `branch` identifies the human-readable comparison side;
- `commit` is the resolved Git commit, not a movable branch name;
- `comparison_run_uuid` is shared by every side of one invocation;
- `run_uuid` uniquely identifies one branch/dtype execution;
- `timestamp_utc` anchors the comparison in time.

Workload shape fields are included alongside these values. This prevents rows for distinct shapes that happen to share an autotune key from being compared as if they represented the same kernel invocation.

CSV writers validate an existing file header before appending. If the on-disk schema does not match the current schema, collection fails instead of silently producing a CSV containing incompatible row formats. Use a new output directory when moving between collector versions.

## Running a comparison

From the repository root, a typical comparison is:

```bash
scripts/vllm/run_unified_attention_3d_solution_comparison.sh \
  main \
  ua-combined-autotune-ARTIFACT
```

The script accepts the same branch/ref and benchmark controls it previously supported. It creates fresh per-branch/per-dtype/run Triton cache directories under `UA_AUTOTUNE_TRITON_CACHE_ROOT`, which defaults to `<output-root>/triton-cache`. The two noise controls can be overridden in the environment when a different stability/runtime tradeoff is desired:

```bash
UA_AUTOTUNE_CV_THRESHOLD=0.01 \
UA_AUTOTUNE_MAX_CV_RETRIES=4 \
scripts/vllm/run_unified_attention_3d_solution_comparison.sh \
  main \
  ua-combined-autotune-ARTIFACT
```

Lowering the CV threshold or increasing retries can improve repeatability, but increases benchmark duration.

## Running as two separate GitHub CI jobs

The branches can also be run independently through the existing `.github/workflows/vllm-benchmarks.yml` workflow and compared after both jobs finish. This is simpler than using the multi-branch comparison runner in CI: each job checks out only its own branch, and no additional Git worktrees or full-history checkout are required.

Commit the collector patch files to both `main-ARTIFACT` and `ua-combined-autotune-ARTIFACT`. Dispatch the existing workflow once for each branch, using the same tag and runner label.

### Install or upgrade the collector

Add the collector preparation as a standalone item in `jobs.build.steps`. Place it immediately after the existing `Install vllm` step and before `Run vllm unified attention bf16`. It must not be nested inside either neighboring step:

```yaml
      - name: Install vllm
        id: install-vllm
        if: ${{ steps.install-benchmarks.outcome == 'success' && !cancelled() }}
        uses: ./.github/actions/install-vllm
        with:
          use-branch-wheel: ${{ inputs.use_branch_wheel }}

      - name: Enable unified-attention artifact collector
        if: ${{ steps.install-vllm.outcome == 'success' && !cancelled() }}
        run: |
          benchmark_file="benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention_benchmark.py"

          if grep -q "AUTOTUNE_CANDIDATES_FILE" "$benchmark_file"; then
            echo "Current collector already present"
          elif grep -q "AUTOTUNE_DECISIONS_FILE" "$benchmark_file"; then
            echo "Upgrading v1 collector to v2"
            git apply scripts/vllm/unified_attention_benchmark_autotune_artifacts_v1_to_v2.patch
          else
            echo "Installing v2 collector"
            git apply scripts/vllm/unified_attention_benchmark_autotune_artifacts.patch
          fi

      - name: Run vllm unified attention bf16
        if: ${{ steps.install-vllm.outcome == 'success' && !cancelled() }}
        run: |
          # Existing BF16 benchmark commands follow.
```

The same preparation step works for both branches:

- an uninstrumented `main-ARTIFACT` receives the complete v2 patch;
- the v1 collector on `ua-combined-autotune-ARTIFACT` receives the v1-to-v2 upgrade;
- a source that already contains the v2 collector is left unchanged.

The collector modifies the benchmark source that `run_benchmark.sh` executes directly, so it can be applied after dependency installation as shown above.

### Pass comparable run metadata

Update both the BF16 and FP8 unified-attention commands to pass provenance, retry settings, and a fresh job-local Triton cache. For BF16:

```bash
UA_AUTOTUNE_BRANCH="$GITHUB_REF_NAME" \
UA_AUTOTUNE_COMMIT="$GITHUB_SHA" \
UA_AUTOTUNE_COMPARISON_RUN_UUID="$TAG" \
UA_AUTOTUNE_CV_THRESHOLD="0.02" \
UA_AUTOTUNE_MAX_CV_RETRIES="2" \
TRITON_CACHE_DIR="$RUNNER_TEMP/triton-cache-$GITHUB_RUN_ID-bf16" \
bash run_benchmark.sh unified_attention --reports "$REPORTS"
```

For FP8, use a separate cache directory and retain the existing `FP8=1` setting:

```bash
UA_AUTOTUNE_BRANCH="$GITHUB_REF_NAME" \
UA_AUTOTUNE_COMMIT="$GITHUB_SHA" \
UA_AUTOTUNE_COMPARISON_RUN_UUID="$TAG" \
UA_AUTOTUNE_CV_THRESHOLD="0.02" \
UA_AUTOTUNE_MAX_CV_RETRIES="2" \
TRITON_CACHE_DIR="$RUNNER_TEMP/triton-cache-$GITHUB_RUN_ID-fp8" \
FP8=1 \
bash run_benchmark.sh unified_attention --reports "$REPORTS"
```

Using the workflow tag as `comparison_run_uuid` is intentional: supply the same unique tag to both dispatches so the independently generated rows can be joined later. The collector creates a distinct `run_uuid` inside each benchmark process.

### Dispatch both branches

For example:

```bash
gh workflow run vllm-benchmarks.yml \
  --ref main-ARTIFACT \
  -f runner_label=max1550 \
  -f tag=ua-autotune-comparison-001

gh workflow run vllm-benchmarks.yml \
  --ref ua-combined-autotune-ARTIFACT \
  -f runner_label=max1550 \
  -f tag=ua-autotune-comparison-001
```

Use a new shared tag for every comparison. Use the same runner label for both jobs and, when possible, the same physical GPU model. Separate jobs are easier to operate but are less controlled than sequential measurements on one physical device, so small timing differences still require repetition.

### Retrieve and compare artifacts

The existing workflow uploads the complete `reports` directory as `benchmark-reports`. The decision and candidate CSVs are written into that directory through the existing `--reports "$REPORTS"` argument, so no additional upload step is needed.

Both workflow runs use the artifact name `benchmark-reports`, but the artifacts belong to different GitHub run IDs. Download them into separate directories before comparison. Match rows using the shared `comparison_run_uuid`, data type, workload shape, autotune key, and the same benchmark variant. Do not require `run_uuid` to match across branches; it identifies one side's benchmark process.

The standard `run_benchmark.sh` executes both the unpatched and tensor-descriptor-patched variants. Compare like with like by filtering on `td_patched` in the decision CSV or `benchmark_name` in both CSVs. Leaving both variants enabled also keeps the workflow result-transformation steps working without further edits.

The workflow continues to run its other configured vLLM and MoE benchmarks unless those steps are disabled in the artifact branches.

## Comparing branches

A reliable comparison should proceed in this order:

1. Match rows with the same comparison run UUID, data type, benchmark variant, workload shape, and autotune key.
2. Confirm the resolved commits and collector schema are the expected ones.
3. Compare candidate membership and statuses before comparing winners.
4. If a candidate is absent from timing results, inspect its pruning status and diagnostic.
5. Compare winner and runner-up timings, CVs, and margins.
6. Treat a winner change with a very small margin or high CV as inconclusive until it reproduces.
7. Use `selection_source`, `cache_hit`, and the presence of timing data to separate fresh tuning, manual-cache, and fixed decisions.

This makes it possible to classify a branch difference as one of the following:

- a different candidate was available;
- the same candidate was pruned differently;
- the same candidates were timed but ranked differently;
- the winner changed within the noise margin;
- a cached or fixed result bypassed fresh tuning.

## Operational notes

- The vLLM source directory used by the runner must not contain unrelated modifications that overlap the temporary patches.
- Use a fresh output directory after changing the CSV schema.
- Pruned candidates legitimately have no timing samples.
- Cache hits legitimately have less timing detail than fresh autotuning.
- Candidate retries are bounded, so noisy hardware or a busy system can still produce a high-CV retained result.

## Validation performed

The collector patches were checked against both `main` and `ua-combined-autotune-ARTIFACT`, the patched Python was compiled, shell syntax was checked, and focused collection smoke tests covered fixed configurations, candidate statuses, CV, rank, and margin fields. No native/compiler rebuild was required because the changes are limited to Python instrumentation, shell orchestration, patch artifacts, and this documentation.
