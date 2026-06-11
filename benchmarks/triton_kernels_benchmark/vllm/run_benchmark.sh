#!/bin/bash
# Run a vllm benchmark before and after applying a local patch (e.g. tensor descriptors).
#
# Usage:
#   bash run_benchmark.sh BENCHMARK_FOLDER [extra args passed to benchmark script...]
#
# BENCHMARK_FOLDER must contain:
#   NAME.patch          - patch to apply to the vllm checkout
#   NAME_benchmark.py   - benchmark script (NAME = basename of BENCHMARK_FOLDER)
#
# Environment variables forwarded to the benchmark script:
#   FP8=1         - enable FP8 configurations
#   DEBUG_BENCH=1 - run only one configuration (faster for sanity checking)
#   PRUNE_CACHE=1 - redirect TRITON_CACHE_DIR to a per-run dir and, after the
#                   benchmark completes, delete every autotune candidate that
#                   wasn't the winner. The surviving cache is saved next to the
#                   benchmark output so its IR (.ttgir/.llir/.spv) can be diffed.
set -e

# Workaround for #6759: BMG OOM after Agama 1222 -> 1249 driver bump.
# expandable_segments alone causes UR_RESULT_ERROR_DEVICE_LOST in CI; pairs with
# torch.xpu.set_per_process_memory_fraction(1.0) called early in the benchmark
# to materialize the virtual segment via the working init path.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

BENCHMARK_FOLDER="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VLLM_DIR="$REPO_ROOT/vllm"

BENCHMARK_DIR="$(cd "$SCRIPT_DIR/$BENCHMARK_FOLDER" && pwd)"
NAME="$(basename "$BENCHMARK_DIR")"
PATCH_FILE="$BENCHMARK_DIR/$NAME.patch"
BENCHMARK_SCRIPT="$BENCHMARK_DIR/${NAME}_benchmark.py"

# Ensure patch is not already applied before baseline
cd "$VLLM_DIR"
if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "=== Reverting previously applied patch ==="
    git apply -R "$PATCH_FILE"
fi

echo "=== Skipping unpatched baseline ==="

echo ""
echo "=== Applying patch ==="
git apply "$PATCH_FILE"

if [[ -n "$PRUNE_CACHE" ]]; then
    if [[ -z "$TRITON_CACHE_DIR" ]]; then
        # No user-supplied cache dir — allocate a per-run one so we never touch ~/.triton/cache.
        TRITON_CACHE_DIR="$BENCHMARK_DIR/triton_cache_$$"
        rm -rf "$TRITON_CACHE_DIR"
        mkdir -p "$TRITON_CACHE_DIR"
        export TRITON_CACHE_DIR
        echo "=== PRUNE_CACHE=1, allocated TRITON_CACHE_DIR=$TRITON_CACHE_DIR ==="
    else
        echo "=== PRUNE_CACHE=1, pruning user-supplied TRITON_CACHE_DIR=$TRITON_CACHE_DIR ==="
    fi
fi

echo ""
echo "=== Running benchmark WITH tensor descriptor patch ==="
TD_PATCHED=1 python "$BENCHMARK_SCRIPT" "$@"

if [[ -n "$PRUNE_CACHE" ]]; then
    echo ""
    echo "=== Pruning autotune losers from $TRITON_CACHE_DIR ==="
    python "$SCRIPT_DIR/prune_autotune_cache.py" "$TRITON_CACHE_DIR" \
        ${PRUNE_KEEP_TOP:+--keep-top "$PRUNE_KEEP_TOP"}
fi

echo ""
echo "=== Reverting patch ==="
git apply -R "$PATCH_FILE"
