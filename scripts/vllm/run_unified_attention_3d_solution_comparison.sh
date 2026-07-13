#!/usr/bin/env bash
# Run unified-attention benchmarks for any number of 3D producer/reducer
# solution branches and write a markdown artifact index plus a comparison plot.
#
# Sister script to run_unified_attention_3d_branch_comparison.sh, which only
# compares two branches (baseline vs active-segments-buffer). This script
# compares all three solution families side by side by default:
#
#   baseline  ua-autotune2-artifact-3d-bench-baseline
#             Decorator autotuning for 2D, 3D tile size still constrained.
#   buffer    ua-3d-active-segments-buffer
#             Producer passes active-segment metadata to the reducer instead of
#             requiring the reducer to reconstruct it from TILE_SIZE.
#   sentinel  ua-3d-tile-autotune-sentinel
#             Unused segment entries are made neutral so the reducer does not
#             need the selected tile size to identify active segments.
#
# After benchmarking it renders scripts/vllm/plot_unified_attention_3d_solution_comparison.py
# into <REPORT_ROOT>/triton_td_solution_comparison-<dtype>.png.
#
# Useful knobs:
#   Positional args           branches to compare. Use either BRANCH or LABEL=BRANCH.
#                             Use baseline=BRANCH to add and select a baseline run.
#                             The run label for baseline=BRANCH is derived from BRANCH.
#   --baseline LABEL          compare all improvement summaries against LABEL.
#   BASELINE_RUN=LABEL        environment equivalent of --baseline.
#   BASELINE_BRANCH=...      baseline branch name
#   BUFFER_BRANCH=...        metadata-buffer branch name
#   SENTINEL_BRANCH=...      sentinel branch name
#   REPORT_ROOT=...          output root; default $REPO_ROOT/tmp/ua-3d-solution-comparison-<timestamp>
#   RUN_BF16=1               run BF16 benchmarks; default 1
#   RUN_FP8=0                run FP8 benchmarks; default 0
#   UA_RUN_MAIN_BENCH=0      run the original broad benchmark sweep; default 0
#   UA_RUN_TARGETED_3D_BENCH=1  run targeted 3D benchmark; default 1
#   MAKE_PLOT=1              render the comparison plot after benchmarking; default 1
#   DEBUG_BENCH=1            pass through to benchmark script for a quick smoke run
#   BENCHMARKING_METHOD=...   default ELAPSED_TIME for local reliability; set UPSTREAM_PYTORCH_PROFILER for CI-style profiling
#   APPLY_PROFILER_FALLBACK_PATCH=1
#                            when using UPSTREAM_PYTORCH_PROFILER, patch XPU profiler extraction for top-level kernel events; default 1
#   APPLY_QSCALED_IGC_WORKAROUND=1
#                            temporarily drop the Q_scaled TD patch hunk that trips an IGC FPE; default 1
#   VLLM_SOURCE=/path/vllm   shared vLLM checkout to copy into worktrees; should be XPU-patched but benchmark-patch-free
#   STRICT_CLEAN_VLLM_SOURCE=1  reject any tracked changes in VLLM_SOURCE; default 0 because XPU-patched vLLM is dirty
#   INSTALL_VLLM=1           force-install vLLM separately in each worktree instead of symlinking VLLM_SOURCE
#   TRANSFORM_RESULTS=1      produce CI-style *-report.csv files; default 0 because local runs may not source capture-hw-details.sh
#   COLLECT_AUTOTUNE_DECISIONS=1
#                            temporarily patch the benchmark to write decision and per-candidate CSVs; default 1
#   UA_AUTOTUNE_CV_THRESHOLD=0.02
#                            retry selected and candidate measurements above this CV; default 0.02
#   UA_AUTOTUNE_MAX_CV_RETRIES=2
#                            maximum retries after the first unstable measurement; default 2
#   UA_AUTOTUNE_TRITON_CACHE_ROOT=...
#                            root for fresh per-branch/per-dtype/run Triton caches; default $REPORT_ROOT/triton-cache
#   UA_BENCH_PATCHED_ONLY=1  skip the unpatched benchmark half inside run_benchmark.sh; default 1
#   DRY_RUN=1                print commands without running benchmarks
set -euo pipefail

BASELINE_BRANCH="${BASELINE_BRANCH:-ua-autotune2-artifact-3d-bench-baseline}"
BUFFER_BRANCH="${BUFFER_BRANCH:-ua-3d-active-segments-buffer}"
SENTINEL_BRANCH="${SENTINEL_BRANCH:-ua-3d-tile-autotune-sentinel}"
RUN_BF16="${RUN_BF16:-1}"
RUN_FP8="${RUN_FP8:-0}"
UA_RUN_MAIN_BENCH="${UA_RUN_MAIN_BENCH:-0}"
UA_RUN_TARGETED_3D_BENCH="${UA_RUN_TARGETED_3D_BENCH:-1}"
MAKE_PLOT="${MAKE_PLOT:-1}"
TRANSFORM_RESULTS="${TRANSFORM_RESULTS:-0}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
DRY_RUN="${DRY_RUN:-0}"
BENCHMARKING_METHOD="${BENCHMARKING_METHOD:-ELAPSED_TIME}"
STRICT_CLEAN_VLLM_SOURCE="${STRICT_CLEAN_VLLM_SOURCE:-0}"
COLLECT_AUTOTUNE_DECISIONS="${COLLECT_AUTOTUNE_DECISIONS:-1}"
UA_AUTOTUNE_CV_THRESHOLD="${UA_AUTOTUNE_CV_THRESHOLD:-0.02}"
UA_AUTOTUNE_MAX_CV_RETRIES="${UA_AUTOTUNE_MAX_CV_RETRIES:-2}"
APPLY_PROFILER_FALLBACK_PATCH="${APPLY_PROFILER_FALLBACK_PATCH:-1}"
APPLY_QSCALED_IGC_WORKAROUND="${APPLY_QSCALED_IGC_WORKAROUND:-1}"
UA_BENCH_PATCHED_ONLY="${UA_BENCH_PATCHED_ONLY:-1}"
BASELINE_RUN="${BASELINE_RUN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REPORT_ROOT="${REPORT_ROOT:-$REPO_ROOT/tmp/ua-3d-solution-comparison-$TIMESTAMP}"
WORKTREE_ROOT="$REPORT_ROOT/worktrees"
SUMMARY_FILE="$REPORT_ROOT/artifacts.md"
VLLM_SOURCE="${VLLM_SOURCE:-$REPO_ROOT/vllm}"
PLOT_SCRIPT="$SCRIPT_DIR/plot_unified_attention_3d_solution_comparison.py"
AUTOTUNE_ARTIFACT_PATCH="$SCRIPT_DIR/unified_attention_benchmark_autotune_artifacts.patch"
AUTOTUNE_ARTIFACT_UPGRADE_PATCH="$SCRIPT_DIR/unified_attention_benchmark_autotune_artifacts_v1_to_v2.patch"
PATCHED_ONLY_RUNNER_PATCH="$SCRIPT_DIR/run_benchmark_patched_only.patch"
PROFILER_FALLBACK_PATCH="$SCRIPT_DIR/upstream_pytorch_profiler_xpu_event_fallback.patch"
AUTOTUNE_TRITON_CACHE_ROOT="${UA_AUTOTUNE_TRITON_CACHE_ROOT:-$REPORT_ROOT/triton-cache}"
COMPARISON_RUN_UUID="${UA_AUTOTUNE_COMPARISON_RUN_UUID:-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')}"
RUN_TIMESTAMP_UTC="${UA_AUTOTUNE_TIMESTAMP_UTC:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
AUTOTUNE_PATCH_USED=""

# Run labels and branches, in plot order. BASELINE_LABEL controls ratio summaries.
LABELS=()
BRANCHES=()
BASELINE_LABEL="$BASELINE_RUN"

mkdir -p "$WORKTREE_ROOT"

run_cmd() {
  {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  } >&2
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

branch_for_label() {
  local label="$1"
  local i
  for i in "${!LABELS[@]}"; do
    if [[ "${LABELS[$i]}" == "$label" ]]; then
      printf '%s' "${BRANCHES[$i]}"
      return
    fi
  done
  echo "Unknown label: $label" >&2
  exit 1
}

sanitize_label() {
  local label="$1"
  label="$(printf '%s' "$label" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^_+//; s/_+$//')"
  if [[ -z "$label" ]]; then
    label="run${#LABELS[@]}"
  fi
  printf '%s' "$label"
}

add_run() {
  local label="$1"
  local branch="$2"
  local existing

  if [[ -z "$label" || -z "$branch" ]]; then
    echo "Run specs must be BRANCH or LABEL=BRANCH, with non-empty values." >&2
    exit 1
  fi

  for existing in "${LABELS[@]}"; do
    if [[ "$existing" == "$label" ]]; then
      echo "Duplicate run label: $label" >&2
      exit 1
    fi
  done

  LABELS+=("$label")
  BRANCHES+=("$branch")

  if [[ "$label" == "baseline" && -z "$BASELINE_LABEL" ]]; then
    BASELINE_LABEL="$label"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [--baseline LABEL] [BRANCH | LABEL=BRANCH]...

Runs unified-attention 3D benchmarks for each branch argument, in order, and
plots them side by side. The baseline for ratio summaries is selected by
--baseline LABEL, BASELINE_RUN=LABEL, baseline=BRANCH, or finally the first run.
The baseline=BRANCH shorthand labels the run from BRANCH, so baseline=main is
displayed as the main run.

If no branch arguments are provided, the default runs are:
  baseline=$BASELINE_BRANCH
  buffer=$BUFFER_BRANCH
  sentinel=$SENTINEL_BRANCH

Examples:
  $(basename "$0") baseline=main ua-dual-tile-autotune ua-3d-tile-autotune-sentinel
  $(basename "$0") main dual=ua-dual-tile-autotune sentinel=ua-3d-tile-autotune-sentinel --baseline main
  $(basename "$0") baseline=ua-autotune2-artifact-3d-bench-baseline dual=ua-dual-tile-autotune
  $(basename "$0") ua-dual-tile-autotune~2 ua-dual-tile-autotune ua-3d-tile-autotune-sentinel

Environment knobs such as RUN_BF16, RUN_FP8, REPORT_ROOT, VLLM_SOURCE,
DEBUG_BENCH, and MAKE_PLOT are unchanged.
EOF
}

parse_run_args() {
  local spec label branch

  if [[ "$#" -eq 0 ]]; then
    add_run baseline "$BASELINE_BRANCH"
    add_run buffer "$BUFFER_BRANCH"
    add_run sentinel "$SENTINEL_BRANCH"
    return
  fi

  while [[ "$#" -gt 0 ]]; do
    spec="$1"
    shift
    case "$spec" in
      -h|--help)
        usage
        exit 0
        ;;
      --baseline)
        if [[ "$#" -eq 0 ]]; then
          echo "--baseline requires a run label" >&2
          exit 1
        fi
        BASELINE_LABEL="$(sanitize_label "$1")"
        shift
        continue
        ;;
      --baseline=*)
        BASELINE_LABEL="$(sanitize_label "${spec#--baseline=}")"
        continue
        ;;
      baseline=*)
        branch="${spec#*=}"
        label="$(sanitize_label "$branch")"
        BASELINE_LABEL="$label"
        ;;
      *=*)
        label="$(sanitize_label "${spec%%=*}")"
        branch="${spec#*=}"
        ;;
      *)
        branch="$spec"
        label="$(sanitize_label "$branch")"
        ;;
    esac
    add_run "$label" "$branch"
  done

  if [[ -z "$BASELINE_LABEL" && "${#LABELS[@]}" -gt 0 ]]; then
    BASELINE_LABEL="${LABELS[0]}"
  fi
}

runs_arg() {
  local IFS=,
  printf '%s' "${LABELS[*]}"
}

validate_baseline_label() {
  local label
  for label in "${LABELS[@]}"; do
    if [[ "$label" == "$BASELINE_LABEL" ]]; then
      return
    fi
  done

  echo "Baseline label '$BASELINE_LABEL' is not one of the configured run labels: $(runs_arg)" >&2
  exit 1
}

require_branch() {
  local branch="$1"
  git -C "$REPO_ROOT" rev-parse --verify --quiet "$branch^{commit}" >/dev/null || {
    echo "Missing branch or ref: $branch" >&2
    exit 1
  }
}

ensure_clean_shared_vllm() {
  if [[ "$INSTALL_VLLM" == "1" ]]; then
    return
  fi
  if [[ ! -d "$VLLM_SOURCE/.git" ]]; then
    cat >&2 <<EOF
No shared vLLM checkout found at: $VLLM_SOURCE
Set VLLM_SOURCE=/path/to/clean/vllm, or rerun with INSTALL_VLLM=1 to install vLLM in each worktree.
EOF
    exit 1
  fi
  if ! git -C "$VLLM_SOURCE" diff --quiet || ! git -C "$VLLM_SOURCE" diff --cached --quiet; then
    if [[ "$STRICT_CLEAN_VLLM_SOURCE" == "1" ]]; then
      cat >&2 <<EOF
The shared vLLM checkout has tracked local modifications: $VLLM_SOURCE
Unset STRICT_CLEAN_VLLM_SOURCE, point VLLM_SOURCE at another checkout, or use INSTALL_VLLM=1.
EOF
      exit 1
    fi
    cat >&2 <<EOF
Warning: shared vLLM checkout has tracked local modifications: $VLLM_SOURCE
This is expected for an XPU-patched vLLM source. Ensure it is free of the branch-specific unified_attention.patch before benchmarking.
Set STRICT_CLEAN_VLLM_SOURCE=1 to reject dirty vLLM sources.
EOF
  fi
}

prepare_worktree() {
  local branch="$1"
  local label="$2"
  local commit
  local wt
  local install_log
  commit="$(git -C "$REPO_ROOT" rev-parse "$branch^{commit}")"
  wt="$WORKTREE_ROOT/$label"
  install_log="$REPORT_ROOT/logs/$label-install-vllm.log"

  run_cmd git -C "$REPO_ROOT" worktree add --detach "$wt" "$commit"

  if [[ "$INSTALL_VLLM" == "1" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      run_cmd ./scripts/vllm/install-vllm.sh --force-reinstall --smoke-test
    else
      mkdir -p "$(dirname "$install_log")"
      echo "Installing vLLM for $label; log: $install_log" >&2
      (
        cd "$wt"
        ./scripts/vllm/install-vllm.sh --force-reinstall --smoke-test
      ) >"$install_log" 2>&1
      if [[ ! -d "$wt/vllm/.git" ]]; then
        echo "vLLM install did not create expected checkout: $wt/vllm" >&2
        echo "Install log: $install_log" >&2
        exit 1
      fi
      echo "Finished vLLM install for $label" >&2
    fi
  else
    echo "Copying shared vLLM source for $label from $VLLM_SOURCE" >&2
    run_cmd cp -a "$VLLM_SOURCE" "$wt/vllm"
  fi
}

apply_qscaled_igc_workaround() {
  local wt="$1"
  local patch_file="$wt/benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch"

  if [[ "$APPLY_QSCALED_IGC_WORKAROUND" != "1" ]]; then
    return
  fi
  if [[ ! -f "$patch_file" ]]; then
    return
  fi
  if ! grep -q "Q_scaled = (Q \* score_scale).to(Q.dtype)" "$patch_file"; then
    return
  fi

  python3 - "$patch_file" <<'PYQSCALED'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
needle = "+        Q_scaled = (Q * score_scale).to(Q.dtype)\n"
pos = text.find(needle)
if pos == -1:
    raise SystemExit(f"Could not find Q_scaled hunk in {path}")
start = text.rfind("\n@@ ", 0, pos)
if start != -1:
    start += 1
elif text.startswith("@@ "):
    start = 0
if start == -1:
    raise SystemExit(f"Could not find Q_scaled hunk header in {path}")
end_marker = "         if USE_SOFTCAP:\n             S = apply_softcap(S, softcap)\n"
end = text.find(end_marker, pos)
if end == -1:
    raise SystemExit(f"Could not find Q_scaled hunk end in {path}")
end += len(end_marker)
path.write_text(text[:start] + text[end:])
PYQSCALED
  echo "Removed Q_scaled TD patch hunk from $patch_file to avoid IGC FPE" >&2
}

apply_profiler_fallback_patch() {
  local wt="$1"
  local benchmark_file="$wt/benchmarks/triton_kernels_benchmark/benchmark_testing.py"

  if [[ "$BENCHMARKING_METHOD" != "UPSTREAM_PYTORCH_PROFILER" || "$APPLY_PROFILER_FALLBACK_PATCH" != "1" ]]; then
    return 1
  fi
  if [[ ! -f "$PROFILER_FALLBACK_PATCH" ]]; then
    echo "Profiler fallback patch not found: $PROFILER_FALLBACK_PATCH" >&2
    exit 1
  fi
  if [[ -f "$benchmark_file" ]] && grep -q "event_device_time_us" "$benchmark_file"; then
    echo "Profiler fallback already present in $benchmark_file" >&2
    return 1
  fi
  if ! git -C "$wt" apply --check "$PROFILER_FALLBACK_PATCH"; then
    echo "Cannot apply profiler fallback patch in $wt" >&2
    exit 1
  fi

  run_cmd git -C "$wt" apply "$PROFILER_FALLBACK_PATCH"
}

revert_profiler_fallback_patch() {
  local wt="$1"

  if [[ "$APPLY_PROFILER_FALLBACK_PATCH" != "1" ]]; then
    return
  fi
  if git -C "$wt" apply --reverse --check "$PROFILER_FALLBACK_PATCH" >/dev/null 2>&1; then
    run_cmd git -C "$wt" apply --reverse "$PROFILER_FALLBACK_PATCH"
  fi
}

apply_autotune_artifact_patch() {
  local wt="$1"
  local benchmark_file="$wt/benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention_benchmark.py"

  if [[ "$COLLECT_AUTOTUNE_DECISIONS" != "1" ]]; then
    return 1
  fi
  if [[ ! -f "$AUTOTUNE_ARTIFACT_PATCH" ]]; then
    echo "Autotune artifact patch not found: $AUTOTUNE_ARTIFACT_PATCH" >&2
    exit 1
  fi
  if git -C "$wt" apply --reverse --check "$AUTOTUNE_ARTIFACT_PATCH" >/dev/null 2>&1; then
    echo "Autotune artifact v2 patch already applied in $wt" >&2
    AUTOTUNE_PATCH_USED="$AUTOTUNE_ARTIFACT_PATCH"
    return 0
  fi
  if [[ -f "$benchmark_file" ]] && grep -q "AUTOTUNE_CANDIDATES_FILE" "$benchmark_file"; then
    echo "Autotune artifact v2 collection already present in $benchmark_file" >&2
    return 1
  fi
  if [[ -f "$benchmark_file" ]] && grep -q "AUTOTUNE_DECISIONS_FILE" "$benchmark_file"; then
    if [[ ! -f "$AUTOTUNE_ARTIFACT_UPGRADE_PATCH" ]]; then
      echo "Autotune artifact v1-to-v2 upgrade patch not found: $AUTOTUNE_ARTIFACT_UPGRADE_PATCH" >&2
      exit 1
    fi
    if ! git -C "$wt" apply --check "$AUTOTUNE_ARTIFACT_UPGRADE_PATCH"; then
      echo "Cannot upgrade embedded autotune artifact collection in $wt" >&2
      exit 1
    fi
    run_cmd git -C "$wt" apply "$AUTOTUNE_ARTIFACT_UPGRADE_PATCH"
    AUTOTUNE_PATCH_USED="$AUTOTUNE_ARTIFACT_UPGRADE_PATCH"
    return 0
  fi
  if ! git -C "$wt" apply --check "$AUTOTUNE_ARTIFACT_PATCH"; then
    echo "Cannot apply autotune artifact patch in $wt" >&2
    exit 1
  fi

  run_cmd git -C "$wt" apply "$AUTOTUNE_ARTIFACT_PATCH"
  AUTOTUNE_PATCH_USED="$AUTOTUNE_ARTIFACT_PATCH"
}

revert_autotune_artifact_patch() {
  local wt="$1"

  if [[ "$COLLECT_AUTOTUNE_DECISIONS" != "1" ]]; then
    return
  fi
  if [[ -n "$AUTOTUNE_PATCH_USED" ]] &&
      git -C "$wt" apply --reverse --check "$AUTOTUNE_PATCH_USED" >/dev/null 2>&1; then
    run_cmd git -C "$wt" apply --reverse "$AUTOTUNE_PATCH_USED"
  fi
  AUTOTUNE_PATCH_USED=""
}

apply_patched_only_runner_patch() {
  local wt="$1"

  if [[ "$UA_BENCH_PATCHED_ONLY" != "1" ]]; then
    return 1
  fi
  if [[ ! -f "$PATCHED_ONLY_RUNNER_PATCH" ]]; then
    echo "Patched-only runner patch not found: $PATCHED_ONLY_RUNNER_PATCH" >&2
    exit 1
  fi
  if git -C "$wt" apply --reverse --check "$PATCHED_ONLY_RUNNER_PATCH" >/dev/null 2>&1; then
    echo "Patched-only runner behavior already present in $wt" >&2
    return 1
  fi
  if ! git -C "$wt" apply --check "$PATCHED_ONLY_RUNNER_PATCH"; then
    echo "Cannot apply patched-only runner patch in $wt" >&2
    exit 1
  fi

  run_cmd git -C "$wt" apply "$PATCHED_ONLY_RUNNER_PATCH"
}

revert_patched_only_runner_patch() {
  local wt="$1"

  if [[ "$UA_BENCH_PATCHED_ONLY" != "1" ]]; then
    return
  fi
  if git -C "$wt" apply --reverse --check "$PATCHED_ONLY_RUNNER_PATCH" >/dev/null 2>&1; then
    run_cmd git -C "$wt" apply --reverse "$PATCHED_ONLY_RUNNER_PATCH"
  fi
}

transform_one() {
  local wt="$1"
  local reports="$2"
  local src_name="$3"
  local dst_name="$4"
  local benchmark_name="$5"
  local src="$reports/$src_name"
  local dst="$reports/$dst_name"

  if [[ "$TRANSFORM_RESULTS" != "1" || ! -f "$src" ]]; then
    return
  fi
  if [[ -z "${GPU_DEVICE:-}" ]]; then
    echo "Skipping transform for $src_name because GPU_DEVICE is unset. Source scripts/capture-hw-details.sh or set TRANSFORM_RESULTS=0." >&2
    return
  fi

  (
    cd "$wt/benchmarks/triton_kernels_benchmark/vllm"
    run_cmd python ../../transform_results.py \
      "$src" \
      "$dst" \
      --tag "$(basename "$reports")" \
      --bgroup "vllm" \
      --benchmark "$benchmark_name" \
      --param_cols "q_heads,k_heads,head_size,qdtype,seq_lens,sliding_window,soft_cap,num_blocks,block_size"
  )
}

transform_reports() {
  local wt="$1"
  local reports="$2"
  local dtype="$3"
  local dtype_suffix=""
  if [[ "$dtype" == "fp8" ]]; then
    dtype_suffix="-fp8"
  fi

  transform_one "$wt" "$reports" \
    "unified-attention-performance.csv" \
    "unified-attention${dtype_suffix}-report.csv" \
    "unified-attn-${dtype}"
  transform_one "$wt" "$reports" \
    "unified-attention-performance-td.csv" \
    "unified-attention${dtype_suffix}-td-report.csv" \
    "unified-attn-${dtype}"
  transform_one "$wt" "$reports" \
    "unified-attention-3d-performance.csv" \
    "unified-attention-3d${dtype_suffix}-report.csv" \
    "unified-attn-3d-${dtype}"
  transform_one "$wt" "$reports" \
    "unified-attention-3d-performance-td.csv" \
    "unified-attention-3d${dtype_suffix}-td-report.csv" \
    "unified-attn-3d-${dtype}"
}

normalize_report_names() {
  local reports="$1"
  local suffix stem src dst

  # Older targeted-3D branches write unified-attention-3d-* files. Mirror them
  # to the canonical unified-attention-* names so every compared run has the
  # same artifact filenames for plotting and transformed reports.
  for suffix in .csv _0.csv _0.png; do
    for stem in performance performance-td; do
      src="$reports/unified-attention-3d-$stem$suffix"
      dst="$reports/unified-attention-$stem$suffix"
      if [[ -f "$src" && ! -f "$dst" ]]; then
        run_cmd cp "$src" "$dst"
      fi
    done
  done
}

run_one_dtype() {
  local wt="$1"
  local label="$2"
  local dtype="$3"
  local reports="$REPORT_ROOT/$label/$dtype"
  local autotune_patch_applied=0
  local patched_only_runner_patch_applied=0
  local profiler_patch_applied=0
  local run_status=0
  local branch
  local commit
  local run_uuid
  local triton_cache_dir
  branch="$(branch_for_label "$label")"
  commit="$(git -C "$REPO_ROOT" rev-parse "$branch^{commit}")"
  run_uuid="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
  triton_cache_dir="$AUTOTUNE_TRITON_CACHE_ROOT/$label/$dtype/$run_uuid"
  mkdir -p "$reports"
  mkdir -p "$triton_cache_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Would run $dtype benchmark for $label in $wt -> $reports (TRITON_CACHE_DIR=$triton_cache_dir)" >&2
    return
  fi

  apply_qscaled_igc_workaround "$wt"

  if apply_profiler_fallback_patch "$wt"; then
    profiler_patch_applied=1
  fi

  if apply_patched_only_runner_patch "$wt"; then
    patched_only_runner_patch_applied=1
  fi

  if apply_autotune_artifact_patch "$wt"; then
    autotune_patch_applied=1
  fi

  set +e
  (
    cd "$wt/benchmarks/triton_kernels_benchmark/vllm"
    if [[ "$dtype" == "fp8" ]]; then
      run_cmd env \
        FP8=1 \
        UA_RUN_MAIN_BENCH="$UA_RUN_MAIN_BENCH" \
        UA_RUN_TARGETED_3D_BENCH="$UA_RUN_TARGETED_3D_BENCH" \
        BENCHMARKING_METHOD="$BENCHMARKING_METHOD" \
        UA_BENCH_PATCHED_ONLY="$UA_BENCH_PATCHED_ONLY" \
        UA_AUTOTUNE_BRANCH="$branch" \
        UA_AUTOTUNE_COMMIT="$commit" \
        UA_AUTOTUNE_COMPARISON_RUN_UUID="$COMPARISON_RUN_UUID" \
        UA_AUTOTUNE_RUN_UUID="$run_uuid" \
        UA_AUTOTUNE_TIMESTAMP_UTC="$RUN_TIMESTAMP_UTC" \
        UA_AUTOTUNE_CV_THRESHOLD="$UA_AUTOTUNE_CV_THRESHOLD" \
        UA_AUTOTUNE_MAX_CV_RETRIES="$UA_AUTOTUNE_MAX_CV_RETRIES" \
        UA_AUTOTUNE_DECISION_REPORTS="$reports" \
        TRITON_CACHE_DIR="$triton_cache_dir" \
        PYTHONPATH="$wt/benchmarks:$wt/vllm:${PYTHONPATH:-}" \
        bash run_benchmark.sh unified_attention --reports "$reports"
    else
      run_cmd env \
        UA_RUN_MAIN_BENCH="$UA_RUN_MAIN_BENCH" \
        UA_RUN_TARGETED_3D_BENCH="$UA_RUN_TARGETED_3D_BENCH" \
        BENCHMARKING_METHOD="$BENCHMARKING_METHOD" \
        UA_BENCH_PATCHED_ONLY="$UA_BENCH_PATCHED_ONLY" \
        UA_AUTOTUNE_BRANCH="$branch" \
        UA_AUTOTUNE_COMMIT="$commit" \
        UA_AUTOTUNE_COMPARISON_RUN_UUID="$COMPARISON_RUN_UUID" \
        UA_AUTOTUNE_RUN_UUID="$run_uuid" \
        UA_AUTOTUNE_TIMESTAMP_UTC="$RUN_TIMESTAMP_UTC" \
        UA_AUTOTUNE_CV_THRESHOLD="$UA_AUTOTUNE_CV_THRESHOLD" \
        UA_AUTOTUNE_MAX_CV_RETRIES="$UA_AUTOTUNE_MAX_CV_RETRIES" \
        UA_AUTOTUNE_DECISION_REPORTS="$reports" \
        TRITON_CACHE_DIR="$triton_cache_dir" \
        PYTHONPATH="$wt/benchmarks:$wt/vllm:${PYTHONPATH:-}" \
        bash run_benchmark.sh unified_attention --reports "$reports"
    fi
  )
  run_status=$?
  set -e

  if [[ "$autotune_patch_applied" == "1" ]]; then
    revert_autotune_artifact_patch "$wt"
  fi
  if [[ "$patched_only_runner_patch_applied" == "1" ]]; then
    revert_patched_only_runner_patch "$wt"
  fi
  if [[ "$profiler_patch_applied" == "1" ]]; then
    revert_profiler_fallback_patch "$wt"
  fi

  if [[ "$run_status" -ne 0 ]]; then
    return "$run_status"
  fi

  normalize_report_names "$reports"
  transform_reports "$wt" "$reports" "$dtype"
}

num_segments_summary() {
  local label="$1"
  if [[ ! -d "$REPORT_ROOT/$label" ]]; then
    printf 'pending'
    return
  fi

  local decision_files=()
  while IFS= read -r file; do
    decision_files+=("$file")
  done < <(find "$REPORT_ROOT/$label" -name 'unified-attention-autotune-decisions.csv' -type f | sort)

  if [[ "${#decision_files[@]}" -eq 0 ]]; then
    printf 'pending'
    return
  fi

  python - "${decision_files[@]}" <<'PYSEGMENTS'
import csv
import re
import sys

values: set[str] = set()
for path in sys.argv[1:]:
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in (
                'selected_NUM_SEGMENTS_PER_SEQ',
                'selected_NUM_SEGMENTS',
                'selected_num_segments',
                'key_NUM_SEGMENTS_PER_SEQ',
                'key_NUM_SEGMENTS',
            ):
                value = (row.get(column) or '').strip()
                if value:
                    values.add(value)
            selected_config = row.get('selected_config') or ''
            for name in ('NUM_SEGMENTS_PER_SEQ', 'NUM_SEGMENTS', 'num_segments'):
                match = re.search(rf'(?:^|;){name}=([^;|]+)', selected_config)
                if match:
                    values.add(match.group(1).strip())

# Non-segment-autotune branches do not record this field because they use the
# fixed 3D value. Keep the report comparable by showing that fixed value.
if not values:
    values.add('16')

def sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)

print(','.join(sorted(values, key=sort_key)))
PYSEGMENTS
}

tile_size_summary() {
  local label="$1"
  if [[ ! -d "$REPORT_ROOT/$label" ]]; then
    printf 'pending'
    return
  fi

  local decision_files=()
  while IFS= read -r file; do
    decision_files+=("$file")
  done < <(find "$REPORT_ROOT/$label" -name 'unified-attention-autotune-decisions.csv' -type f | sort)

  if [[ "${#decision_files[@]}" -eq 0 ]]; then
    printf 'pending'
    return
  fi

  python - "${decision_files[@]}" <<'PYTILES'
import csv
import sys

values: set[str] = set()
for path in sys.argv[1:]:
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (row.get('selected_TILE_SIZE') or '').strip()
            if value:
                values.add(value)

if not values:
    print('n/a')
else:
    def sort_key(value: str):
        return (0, int(value)) if value.isdigit() else (1, value)

    print(','.join(sorted(values, key=sort_key)))
PYTILES
}

make_plot() {
  if [[ "$MAKE_PLOT" != "1" ]]; then
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Would render plot via $PLOT_SCRIPT" >&2
    return
  fi
  if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "Plot script not found: $PLOT_SCRIPT (skipping plot)" >&2
    return
  fi

  local dtypes=()
  [[ "$RUN_BF16" == "1" ]] && dtypes+=(bf16)
  [[ "$RUN_FP8" == "1" ]] && dtypes+=(fp8)

  for dtype in "${dtypes[@]}"; do
    local out="$REPORT_ROOT/triton_td_solution_comparison-$dtype.png"
    echo "Rendering plot for $dtype -> $out" >&2
    if ! run_cmd python "$PLOT_SCRIPT" \
      --root "$REPORT_ROOT" \
      --dtype "$dtype" \
      --runs "$(runs_arg)" \
      --baseline "$BASELINE_LABEL" \
      --out "$out"; then
      echo "Plot rendering failed for $dtype (continuing)" >&2
    fi
  done
}

write_summary() {
  {
    echo "# Unified Attention 3D Solution Comparison Artifacts"
    echo
    echo "- Report root: [$REPORT_ROOT]($REPORT_ROOT)"
    echo "- Comparison run UUID: \`$COMPARISON_RUN_UUID\`"
    echo "- Timestamp (UTC): \`$RUN_TIMESTAMP_UTC\`"
    echo "- Run labels: \`$(runs_arg)\`"
    echo "- Baseline run: \`$BASELINE_LABEL\` (\`$(branch_for_label "$BASELINE_LABEL")\`)"
    echo "- UA_RUN_MAIN_BENCH: \`$UA_RUN_MAIN_BENCH\`"
    echo "- UA_RUN_TARGETED_3D_BENCH: \`$UA_RUN_TARGETED_3D_BENCH\`"
    echo "- RUN_BF16: \`$RUN_BF16\`"
    echo "- RUN_FP8: \`$RUN_FP8\`"
    echo "- BENCHMARKING_METHOD: \`$BENCHMARKING_METHOD\`"
    echo "- APPLY_PROFILER_FALLBACK_PATCH: \`$APPLY_PROFILER_FALLBACK_PATCH\`"
    echo "- APPLY_QSCALED_IGC_WORKAROUND: \`$APPLY_QSCALED_IGC_WORKAROUND\`"
    echo "- TRANSFORM_RESULTS: \`$TRANSFORM_RESULTS\`"
    echo "- COLLECT_AUTOTUNE_DECISIONS: \`$COLLECT_AUTOTUNE_DECISIONS\`"
    echo "- UA_AUTOTUNE_CV_THRESHOLD: \`$UA_AUTOTUNE_CV_THRESHOLD\`"
    echo "- UA_AUTOTUNE_MAX_CV_RETRIES: \`$UA_AUTOTUNE_MAX_CV_RETRIES\`"
    echo "- UA_AUTOTUNE_TRITON_CACHE_ROOT: \`$AUTOTUNE_TRITON_CACHE_ROOT\`"
    echo
    if [[ "$MAKE_PLOT" == "1" ]]; then
      echo "## Plots"
      echo
      for png in "$REPORT_ROOT"/triton_td_solution_comparison-*.png; do
        [[ -e "$png" ]] || continue
        echo "- [$(basename "$png")]($png)"
      done
      echo
    fi
    echo "## Branches"
    echo
    echo "| Label | Branch | Commit |"
    echo "|---|---|---|"
    for label in "${LABELS[@]}"; do
      echo "| $label | \`$(branch_for_label "$label")\` | \`$(git -C "$REPO_ROOT" rev-parse --short=12 "$(branch_for_label "$label")^{commit}")\` |"
    done
    echo
    echo "## Solution Summary"
    echo
    echo "| Label | Branch | Selected TILE_SIZE | Num segments |"
    echo "|---|---|---|---:|"
    for label in "${LABELS[@]}"; do
      echo "| $label | \`$(branch_for_label "$label")\` | $(tile_size_summary "$label") | $(num_segments_summary "$label") |"
    done
    echo
    for label in "${LABELS[@]}"; do
      echo "## $label"
      if [[ -d "$REPORT_ROOT/$label" ]]; then
        for dtype_dir in "$REPORT_ROOT/$label"/*; do
          [[ -d "$dtype_dir" ]] || continue
          echo
          echo "### $(basename "$dtype_dir")"
          echo "- Directory: [$dtype_dir]($dtype_dir)"
          while IFS= read -r file; do
            echo "- [$(basename "$file")]($file)"
          done < <(find "$dtype_dir" -maxdepth 1 -type f | sort)
        done
      else
        echo "No artifacts found."
      fi
      echo
    done
  } > "$SUMMARY_FILE"

  echo
  echo "Artifact summary: $SUMMARY_FILE"
  echo "Report root: $REPORT_ROOT"
}

main() {
  parse_run_args "$@"
  validate_baseline_label

  local label
  for label in "${LABELS[@]}"; do
    require_branch "$(branch_for_label "$label")"
  done
  ensure_clean_shared_vllm

  for label in "${LABELS[@]}"; do
    prepare_worktree "$(branch_for_label "$label")" "$label"
  done

  if [[ "$RUN_BF16" == "1" ]]; then
    for label in "${LABELS[@]}"; do
      run_one_dtype "$WORKTREE_ROOT/$label" "$label" bf16
    done
  fi

  if [[ "$RUN_FP8" == "1" ]]; then
    for label in "${LABELS[@]}"; do
      run_one_dtype "$WORKTREE_ROOT/$label" "$label" fp8
    done
  fi

  make_plot
  write_summary
}

main "$@"
