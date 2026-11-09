#!/usr/bin/env bash
# Local driver: run the 5 vLLM benchmark invocations from vllm-benchmarks.yml.
# Each runs baseline (TD_PATCHED=0) then tensor-descriptor-patched (TD_PATCHED=1).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# capture-hw-details is required, else report generation crashes.
source "$ROOT/scripts/capture-hw-details.sh" > /dev/null 2>&1

REPORTS="$ROOT/reports/vllm-benchmarks"
LOGS="$REPORTS/logs"
mkdir -p "$LOGS"
TAG="local"

SUMMARY="$REPORTS/SUMMARY.txt"
: > "$SUMMARY"
echo "vLLM benchmark run started: $(date)" | tee -a "$SUMMARY"
echo "GPU=$GPU_DEVICE torch=$TORCH_VERSION" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

cd "$ROOT/benchmarks/triton_kernels_benchmark/vllm"

# name:fp8flag:label  (matches the 5 CI steps)
run_bench() {
  local name="$1" fp8="$2" label="$3"
  local log="$LOGS/${label}.log"
  echo ">>> [$(date +%H:%M:%S)] Benchmark: $label (name=$name FP8=$fp8)" | tee -a "$SUMMARY"
  local start=$(date +%s)
  FP8="$fp8" bash run_benchmark.sh "$name" --reports "$REPORTS" --tag "$TAG" > "$log" 2>&1
  local rc=$?
  local end=$(date +%s)
  echo "    exit=$rc dur=$((end-start))s log=$log" | tee -a "$SUMMARY"
}

run_bench unified_attention 0 unified_attention_bf16
run_bench unified_attention 1 unified_attention_fp8
run_bench batched_moe        0 batched_moe_bf16
run_bench batched_moe        1 batched_moe_fp8
run_bench fused_moe          0 fused_moe_bf16

echo "" | tee -a "$SUMMARY"
echo "vLLM benchmark run finished: $(date)" | tee -a "$SUMMARY"
echo "ALL_BENCHMARKS_DONE"
