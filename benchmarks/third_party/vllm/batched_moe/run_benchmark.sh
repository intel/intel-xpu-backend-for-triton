#!/bin/bash
# Run batched_moe benchmark before and after applying tensor descriptor patch
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VLLM_DIR="$REPO_ROOT/vllm"
TARGET_FILE="vllm/model_executor/layers/fused_moe/fused_batched_moe.py"

# The basic patch already contains changes to this file, so we skip this check for now.
# Check that target file has no uncommitted changes
# cd "$VLLM_DIR"
# if ! git diff --quiet "$TARGET_FILE"; then
#     echo "Error: $TARGET_FILE has uncommitted changes."
#     echo "Run: cd $VLLM_DIR && git checkout $TARGET_FILE"
#     exit 1
# fi

echo "=== Running benchmark WITHOUT patch ==="
TD_PATCHED=0 python "$SCRIPT_DIR/batched_moe_benchmark.py" "$@"

echo ""
echo "=== Applying patch ==="
cd "$VLLM_DIR"
git apply "$SCRIPT_DIR/batched_moe.patch"

echo ""
echo "=== Running benchmark WITH tensor descriptor patch ==="
TD_PATCHED=1 python "$SCRIPT_DIR/batched_moe_benchmark.py" "$@"

# echo ""
# echo "=== Reverting patch ==="
# git apply -R "$SCRIPT_DIR/batched_moe.patch"
