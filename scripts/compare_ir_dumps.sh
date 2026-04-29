#!/bin/bash
# Helper script to compare IR dumps between branches
#
# Usage: ./scripts/compare_ir_dumps.sh

set -e

MAIN_DIR="/tmp/ir_dump_main"
PINUPDATE_DIR="/tmp/ir_dump_pin-update"

if [ ! -d "$MAIN_DIR" ] || [ ! -d "$PINUPDATE_DIR" ]; then
    echo "ERROR: IR dumps not found!"
    echo ""
    echo "Please run the following first:"
    echo "  1. git checkout main"
    echo "  2. ./scripts/debug_tdesc_slowdown.sh main"
    echo "  3. git checkout quinnlp/pin-update"
    echo "  4. ./scripts/debug_tdesc_slowdown.sh pin-update"
    echo ""
    exit 1
fi

echo "==================================="
echo "Comparing IR between branches"
echo "==================================="
echo ""

echo "1. Layout conversion counts:"
echo "   main:       $(grep -c "tt.convert_layout\|ttg.convert_layout" "$MAIN_DIR/full_output.log" || echo 0)"
echo "   pin-update: $(grep -c "tt.convert_layout\|ttg.convert_layout" "$PINUPDATE_DIR/full_output.log" || echo 0)"
echo ""

echo "2. Checking for differences after RemoveLayoutConversions:"
if diff -q "$MAIN_DIR/ttgir_after_remove_layout_conversions.mlir" "$PINUPDATE_DIR/ttgir_after_remove_layout_conversions.mlir" > /dev/null 2>&1; then
    echo "   ✓ IDENTICAL"
else
    echo "   ✗ DIFFERENT"
    echo ""
    echo "   View diff with:"
    echo "   diff -u $MAIN_DIR/ttgir_after_remove_layout_conversions.mlir $PINUPDATE_DIR/ttgir_after_remove_layout_conversions.mlir | less"
fi
echo ""

echo "3. Checking for differences after MaterializeBlockPointer:"
if diff -q "$MAIN_DIR/ttgir_after_materialize_block_pointer.mlir" "$PINUPDATE_DIR/ttgir_after_materialize_block_pointer.mlir" > /dev/null 2>&1; then
    echo "   ✓ IDENTICAL"
else
    echo "   ✗ DIFFERENT"
    echo ""
    echo "   View diff with:"
    echo "   diff -u $MAIN_DIR/ttgir_after_materialize_block_pointer.mlir $PINUPDATE_DIR/ttgir_after_materialize_block_pointer.mlir | less"
fi
echo ""

echo "4. Checking for differences in LLVM IR:"
if diff -q "$MAIN_DIR/llir.mlir" "$PINUPDATE_DIR/llir.mlir" > /dev/null 2>&1; then
    echo "   ✓ IDENTICAL"
else
    echo "   ✗ DIFFERENT"
    echo ""
    echo "   View diff with:"
    echo "   diff -u $MAIN_DIR/llir.mlir $PINUPDATE_DIR/llir.mlir | less"
fi
echo ""

echo "==================================="
echo "Key things to look for in the diffs:"
echo "  - Extra tt.convert_layout operations"
echo "  - Different encoding attributes"
echo "  - Different load/store patterns"
echo "  - Different block_io attributes"
echo "==================================="
