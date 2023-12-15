#!/usr/bin/env bash

# Given an initial Triton MLIR as input, this script generates MLIR of a specified target (e.g., optimized-ttir, ttgir, optimized-ttgir, llir), using the same pipeline specified in compiler.py. Additional compiler options can be passed for debugging purposes (e.g., -mlir-print-ir-after-all).
# Example usage:
# ./triton-opt.sh --triton-opt=<path to triton-opt> 01-vector-add-xpu.ttir --target=llir -mlir-print-ir-after-all

for i in "$@"; do
  case $i in
    --triton-opt=*)
      TRITON_OPT="${i#*=}"
      shift # past argument=value
      ;;
    --target=*)
      TARGET="${i#*=}"
      shift # past argument=value
      ;;
    --help)
      echo "Example usage: ./triton-opt.sh --triton-opt=<path to triton-opt> 01-vector-add-xpu.ttir --target=llir -mlir-print-ir-after-all"
      exit 1
      ;;
    *)
      ARGS+="${i} "
      shift
      ;;
  esac
done

pipeline=""
case $TARGET in
  "llir")
    pipeline="-convert-scf-to-cf -convert-index-to-llvm -convert-triton-gpu-to-llvm -convert-arith-to-llvm -canonicalize -cse -symbol-dce "$pipeline
    ;&
  "optimized-ttgir")
    pipeline="-tritongpu-coalesce -tritongpu-remove-layout-conversions -tritongpu-accelerate-matmul -tritongpu-remove-layout-conversions -tritongpu-optimize-dot-operands -tritongpu-pipeline -tritongpu-prefetch -tritongpu-optimize-dot-operands -tritongpu-remove-layout-conversions -tritongpu-decompose-conversions -tritongpu-reorder-instructions -cse -symbol-dce "$pipeline
    ;&
  "ttgir")
    pipeline="-convert-triton-to-tritongpu "$pipeline
    ;&
  "optimized-ttir")
    pipeline="-inline -inline -triton-combine -canonicalize -cse -loop-invariant-code-motion -symbol-dce "$pipeline
    ;;
  *)
    echo "Invalid target: "$TARGET
    exit 1
    ;;
esac

echo "Running "$TRITON_OPT $pipeline $ARGS
$TRITON_OPT $pipeline $ARGS
