#!/bin/bash
# Script to dump IR for comparing tensor descriptor performance between branches
#
# Usage:
#   1. Run on main branch: ./scripts/debug_tdesc_slowdown.sh main
#   2. Run on pin-update branch: ./scripts/debug_tdesc_slowdown.sh pin-update
#   3. Compare: diff -u /tmp/ir_dump_main/ /tmp/ir_dump_pin-update/

set -e

BRANCH_NAME="${1:-unknown}"
OUTPUT_DIR="/tmp/ir_dump_${BRANCH_NAME}"

echo "==================================="
echo "Dumping IR for branch: $BRANCH_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "==================================="

# Clean output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Set environment variables for MLIR dumps
export MLIR_ENABLE_DUMP=1
export DEBUG_BENCH=1  # Run only first config
export TD_PATCHED=1   # Apply tensor descriptor patch

# Redirect stderr to capture MLIR dumps
cd benchmarks/triton_kernels_benchmark/vllm

echo ""
echo "Running unified_attention benchmark with IR dumps..."
echo "(This will take a minute, running one config only)"
echo ""

bash run_benchmark.sh unified_attention --reports "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/full_output.log"

echo ""
echo "==================================="
echo "Extracting key IR stages..."
echo "==================================="

# Extract TTGIR after key passes
grep -A 200 "IR Dump After.*RemoveLayoutConversions" "$OUTPUT_DIR/full_output.log" > "$OUTPUT_DIR/ttgir_after_remove_layout_conversions.mlir" || echo "RemoveLayoutConversions not found"
grep -A 200 "IR Dump After.*MaterializeBlockPointer" "$OUTPUT_DIR/full_output.log" > "$OUTPUT_DIR/ttgir_after_materialize_block_pointer.mlir" || echo "MaterializeBlockPointer not found"
grep -A 200 "IR Dump After.*OptimizeDotOperands" "$OUTPUT_DIR/full_output.log" > "$OUTPUT_DIR/ttgir_after_optimize_dot_operands.mlir" || echo "OptimizeDotOperands not found"

# Extract LLIR
grep -A 500 "IR Dump After.*ConvertTritonIntelGPUToLLVM\|IR Dump After.*ConvertTritonGPUToLLVM" "$OUTPUT_DIR/full_output.log" > "$OUTPUT_DIR/llir.mlir" || echo "LLVM lowering not found"

# Count layout conversions
echo ""
echo "Layout conversion stats:"
grep -c "tt.convert_layout\|ttg.convert_layout" "$OUTPUT_DIR/full_output.log" || echo "0 layout conversions"

echo ""
echo "==================================="
echo "IR dump complete!"
echo "Files saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - ttgir_after_materialize_block_pointer.mlir"
echo "  - ttgir_after_remove_layout_conversions.mlir"
echo "  - llir.mlir"
echo "  - full_output.log (complete output)"
echo ""
echo "To compare with other branch:"
echo "  diff -u /tmp/ir_dump_main/ttgir_after_remove_layout_conversions.mlir /tmp/ir_dump_pin-update/ttgir_after_remove_layout_conversions.mlir | less"
echo "==================================="
