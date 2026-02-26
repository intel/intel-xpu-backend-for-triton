#!/bin/bash
# Run batched_moe benchmark before and after applying tensor descriptor patch
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VLLM_DIR="$REPO_ROOT/vllm"
PATCH_FILE="$SCRIPT_DIR/batched_moe.patch"

# Ensure patch is not already applied before baseline
cd "$VLLM_DIR"
if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "=== Reverting previously applied patch ==="
    git apply -R "$PATCH_FILE"
fi

echo "=== Running benchmark WITHOUT patch ==="
TD_PATCHED=0 python "$SCRIPT_DIR/batched_moe_benchmark.py" "$@"

echo ""
echo "=== Applying patch ==="
git apply "$PATCH_FILE"

echo ""
echo "=== Running benchmark WITH tensor descriptor patch ==="
TD_PATCHED=1 python "$SCRIPT_DIR/batched_moe_benchmark.py" "$@"

echo ""
echo "=== Reverting patch ==="
git apply -R "$PATCH_FILE"
