#!/bin/bash
# EXPERIMENT (#8742, not for merge): run the REAL benchmark CLI on subsets of the unified attention configs to find
# out whether the bad state needs the earlier configs of the same process.
set +e
cd "$GITHUB_WORKSPACE/vllm" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
run() {  # label, UA_ONLY value
  UA_ONLY="$2" TD_PATCHED=0 triton-benchmarks run vllm-unified-attn-bf16 --reports "$REPORTS" --tag "$1" > "/tmp/ua_$1.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "--- tail of failing log $1 (exit $rc) ---"; tail -n 25 "/tmp/ua_$1.log"; echo "--- end ---"; fi
  echo "$1 (UA_ONLY='$2'): exit $rc | $(grep -oE 'Mismatched elements: [0-9]+ / [0-9]+' "/tmp/ua_$1.log" | head -1) | $(grep -oE 'triton(-td)? to torch \[[^]]{0,60}' "/tmp/ua_$1.log" | head -1)"
}
which triton-benchmarks; triton-benchmarks run -h 2>&1 | head -5; pwd
echo "=== A: the failing config alone, 8 fresh processes ==="
for i in 1 2 3 4 5 6 7 8; do run "alone-$i" alone; done
echo "=== B: all configs of the failing model (64-8-128, no sliding window) ==="
run model model
echo "=== C: full run (control; this is what normally fails) ==="
run full ""
