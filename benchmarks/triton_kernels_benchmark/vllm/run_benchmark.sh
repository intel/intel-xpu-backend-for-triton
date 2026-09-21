#!/bin/bash
# Run a vllm benchmark before and after applying a local patch (e.g. tensor descriptors).
#
# Usage:
#   bash run_benchmark.sh BENCHMARK_FOLDER [extra args passed to benchmark script...]
#
# BENCHMARK_FOLDER must contain:
#   NAME_benchmark.py   - benchmark script (NAME = basename of BENCHMARK_FOLDER)
#   and either:
#     NAME.patch        - single patch: run baseline (TD off) then patched (TD on), or
#     N-*.patch files   - an incremental patch series: each patch is swept as its own
#                         provider (see "Patch-series sweep" below).
#
# Patch-series sweep (folders containing N-*.patch files, e.g. unified_attention):
#   Each numbered patch is applied on its own and benchmarked as a distinct provider,
#   labelled by the patch basename (PROVIDER_LABEL) so runs don't collide. Patches
#   0 and 1 are the baselines and run in both tensor-descriptor modes (use_td=False
#   and use_td=True); every other patch runs with tensor descriptors on (use_td=True).
#   The reference NAME.patch is never applied. The pytorch/sycl-tla providers do not
#   depend on the patched kernel, so they are benchmarked once up front and each
#   per-patch run selects only its own triton provider (no redundant re-timing).
#
# Environment variables forwarded to the benchmark script:
#   FP8=1         - enable FP8 configurations
#   DEBUG_BENCH=1 - run only one configuration (faster for sanity checking)
set -e

BENCHMARK_FOLDER="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VLLM_DIR="$REPO_ROOT/vllm"

BENCHMARK_DIR="$(cd "$SCRIPT_DIR/$BENCHMARK_FOLDER" && pwd)"
NAME="$(basename "$BENCHMARK_DIR")"
PATCH_FILE="$BENCHMARK_DIR/$NAME.patch"

# triton-benchmarks CLI key: vllm-<folder with dashes>, plus -fp8 when FP8 configs are requested.
KEY="vllm-${NAME//_/-}"
if [ "${FP8:-0}" = "1" ]; then
    case "$NAME" in
        unified_attention | batched_moe) KEY="$KEY-fp8" ;;
        *)
            echo "Error: FP8=1 is not supported for benchmark '$NAME' (no '$KEY-fp8' CLI config registered)." >&2
            exit 1
            ;;
    esac
fi

cd "$VLLM_DIR"

# The moe benchmarks import vLLM's source-tree test helpers (`from tests.kernels...`),
# which are not part of the installed vllm package. Put the vllm checkout on PYTHONPATH
# so those imports resolve when the benchmark runs from the installed CLI wheel.
export PYTHONPATH="$VLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Collect the incremental patch series (N-*.patch), numerically sorted. Excludes the
# reference NAME.patch, which never starts with a digit.
mapfile -t SERIES_PATCHES < <(find "$BENCHMARK_DIR" -maxdepth 1 -name '[0-9]*-*.patch' -printf '%f\n' | sort -V)

# ---------------------------------------------------------------------------
# Single-patch mode (no numbered series): baseline then patched, as before.
# ---------------------------------------------------------------------------
if [ "${#SERIES_PATCHES[@]}" -eq 0 ]; then
    if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
        echo "=== Reverting previously applied patch ==="
        git apply -R "$PATCH_FILE"
    fi

    echo "=== Running benchmark WITHOUT patch ==="
    TD_PATCHED=0 triton-benchmarks run "$KEY" "$@"

    echo ""
    echo "=== Applying patch ==="
    git apply "$PATCH_FILE"

    echo ""
    echo "=== Running benchmark WITH tensor descriptor patch ==="
    TD_PATCHED=1 triton-benchmarks run "$KEY" "$@"

    echo ""
    echo "=== Reverting patch ==="
    git apply -R "$PATCH_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Patch-series sweep: one provider per patch (baselines 0 and 1 get both TD modes).
# ---------------------------------------------------------------------------

# Revert whatever patch is currently applied (best effort) so the tree is clean on exit.
CURRENT_PATCH=""
cleanup() {
    if [ -n "$CURRENT_PATCH" ] && git apply -R --check "$CURRENT_PATCH" 2>/dev/null; then
        echo "=== Reverting $(basename "$CURRENT_PATCH") ==="
        git apply -R "$CURRENT_PATCH"
    fi
    CURRENT_PATCH=""
}
trap cleanup EXIT

# Start from a clean tree: undo any series patch left applied by a previous run.
for f in "${SERIES_PATCHES[@]}"; do
    if git apply -R --check "$BENCHMARK_DIR/$f" 2>/dev/null; then
        echo "=== Reverting previously applied $f ==="
        git apply -R "$BENCHMARK_DIR/$f"
    fi
done

# The pytorch and sycl-tla providers don't use the (patched) triton kernel, so their
# results are identical for every patch. Benchmark them once on the clean tree; the
# per-patch runs below select only their own triton provider via --provider.
#
# Each reference provider runs in its OWN best-effort invocation: they are only
# comparison baselines, so a failure must NOT abort the triton patch sweep below.
# (e.g. sycl-tla lacks a compiled kernel for some configs and its fallback path can
# OOM — a vllm-xpu-kernels limitation, unrelated to the triton kernel under test.)
# A distinct PROVIDER_LABEL per provider keeps their plots/reports from colliding.
ref_providers=(pytorch)
if [ "${FP8:-0}" != "1" ]; then
    ref_providers+=(sycl-tla)
fi
for rp in "${ref_providers[@]}"; do
    echo ""
    echo "=== Running reference provider: $rp (once, best-effort) ==="
    TD_PATCHED=0 PROVIDER_LABEL="$rp" triton-benchmarks run "$KEY" --provider "$rp" "$@" \
        || echo "WARN: reference provider '$rp' failed; continuing without it"
done

# The pristine (unpatched) upstream triton kernel, TD off: the original 'triton'
# baseline. Runs here on the clean tree (no patch applied), best-effort like the
# other baselines. Unlabelled so it is the classic 'triton' provider.
echo ""
echo "=== Running baseline provider: triton (unpatched, TD off) ==="
TD_PATCHED=0 triton-benchmarks run "$KEY" --provider triton "$@" \
    || echo "WARN: baseline provider 'triton' failed; continuing without it"

for f in "${SERIES_PATCHES[@]}"; do
    label="${f%.patch}"          # provider label, e.g. 0-no-non-pass
    num="${label%%-*}"           # leading patch number

    # Baselines 0 and 1 run both TD modes; every other patch runs with TD on only.
    if [ "$num" = "0" ] || [ "$num" = "1" ]; then
        td_modes=(0 1)
    else
        td_modes=(1)
    fi

    echo ""
    echo "=== Applying $f ==="
    CURRENT_PATCH="$BENCHMARK_DIR/$f"
    git apply "$CURRENT_PATCH"

    for td in "${td_modes[@]}"; do
        # Provider name must match the key built in the benchmark:
        # triton[-td]-<label>. Select only it so pytorch/sycl-tla aren't re-timed.
        prov="triton"
        [ "$td" = "1" ] && prov="triton-td"
        prov="$prov-$label"
        echo ""
        echo "=== Running $prov (TD_PATCHED=$td) ==="
        TD_PATCHED="$td" PROVIDER_LABEL="$label" triton-benchmarks run "$KEY" --provider "$prov" "$@"
    done

    echo ""
    echo "=== Reverting $f ==="
    git apply -R "$CURRENT_PATCH"
    CURRENT_PATCH=""
done

# Finally, benchmark the reference patch (NAME.patch — the full optimization, the
# de-facto endpoint of the series) as the classic unlabelled 'triton-td' provider,
# TD on. The patch file itself is not modified, only applied and reverted.
if [ -f "$PATCH_FILE" ]; then
    # Undo any leftover application from a previous crashed run.
    if git apply -R --check "$PATCH_FILE" 2>/dev/null; then git apply -R "$PATCH_FILE"; fi
    echo ""
    echo "=== Applying reference patch $(basename "$PATCH_FILE") ==="
    CURRENT_PATCH="$PATCH_FILE"
    git apply "$CURRENT_PATCH"

    echo ""
    echo "=== Running triton-td (reference / full optimization, TD_PATCHED=1) ==="
    TD_PATCHED=1 triton-benchmarks run "$KEY" --provider triton-td "$@"

    echo ""
    echo "=== Reverting reference patch ==="
    git apply -R "$CURRENT_PATCH"
    CURRENT_PATCH=""
fi
