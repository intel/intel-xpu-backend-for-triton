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

# Ensure patch is not already applied before baseline
cd "$VLLM_DIR"
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
