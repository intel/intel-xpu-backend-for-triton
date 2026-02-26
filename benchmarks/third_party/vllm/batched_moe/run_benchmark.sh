#!/bin/bash
# Run batched_moe benchmark before and after applying tensor descriptor patch
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VLLM_DIR="$REPO_ROOT/vllm"
TARGET_FILE="vllm/model_executor/layers/fused_moe/fused_batched_moe.py"

echo "=== Running benchmark WITHOUT patch ==="
TD_PATCHED=0 python "$SCRIPT_DIR/batched_moe_benchmark.py" "$@"

echo ""
echo "=== Applying patch ==="
cd "$VLLM_DIR"
git apply "$SCRIPT_DIR/batched_moe.patch"

echo ""
echo "=== Running benchmark WITH tensor descriptor patch ==="
TD_PATCHED=1 python "$SCRIPT_DIR/batched_moe_benchmark.py" "$@"

echo ""
echo "=== Reverting patch ==="
git apply -R "$SCRIPT_DIR/batched_moe.patch"
