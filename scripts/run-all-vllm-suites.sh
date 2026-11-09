#!/usr/bin/env bash
# Local driver: run every vLLM test suite from vllm-tests-reusable.yml sequentially
# (single GPU), isolated + continue-on-failure, mirroring the CI matrix.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPORTS="$ROOT/reports/vllm"
LOGS="$REPORTS/logs"
mkdir -p "$LOGS"

SKIPLIST="$ROOT/scripts/skiplist/default"
BASE_CMD=(bash scripts/test-triton.sh --skip-pip-install --skip-pytorch-install
          --reports-dir "$REPORTS" --skip-list "$SKIPLIST")

# CI matrix order (vllm-rest expanded to gdn-attn/linear-attn/deepgemm).
# Fast suites first to validate the harness end-to-end.
SUITES=(
  linear-attn
  deepgemm
  quant
  gdn-attn
  triton-attn
  mamba
  kda
  moe
  tdesc
  spec-decode
  mrv2
)

SUMMARY="$REPORTS/SUMMARY.txt"
: > "$SUMMARY"
echo "vLLM suite run started: $(date)" | tee -a "$SUMMARY"
echo "torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null) triton=$(python -c 'import triton;print(triton.__version__)' 2>/dev/null)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for suite in "${SUITES[@]}"; do
  log="$LOGS/${suite}.log"
  echo ">>> [$(date +%H:%M:%S)] Running suite: $suite" | tee -a "$SUMMARY"
  start=$(date +%s)
  "${BASE_CMD[@]}" "--vllm-${suite}" > "$log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "    exit=$rc dur=$((end-start))s log=$log" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "vLLM suite run finished: $(date)" | tee -a "$SUMMARY"
echo "ALL_SUITES_DONE"
