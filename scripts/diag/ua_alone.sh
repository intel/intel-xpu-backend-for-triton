#!/bin/bash
# EXPERIMENT (#8742, not for merge): does the bad process state occur per JOB or per PROCESS, and does the failing
# config ever fail when run ALONE in a fresh process during a job where the FULL run fails?
# Interleaves alone/full processes so one job gives several independent samples of each.
set +e
source "$GITHUB_WORKSPACE/scripts/capture-hw-details.sh" > /dev/null 2>&1   # needed by the CLI's report step
cd "$GITHUB_WORKSPACE/vllm" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
run() {  # label, UA_ONLY value
  local t0=$SECONDS
  UA_ONLY="$2" TD_PATCHED=0 triton-benchmarks run vllm-unified-attention --reports "$REPORTS" --tag "$1" > "/tmp/ua_$1.log" 2>&1
  local rc=$?
  local sig verdict
  sig=$(grep -oE 'Mismatched elements: [0-9]+ / [0-9]+' "/tmp/ua_$1.log" | head -1)
  if [ -n "$sig" ]; then verdict=MISMATCH; elif [ $rc -eq 0 ]; then verdict=PASS; else verdict="OTHER-FAIL(exit $rc)"; fi
  echo "RESULT $1 (UA_ONLY='$2'): $verdict | ${sig:--} | $((SECONDS - t0))s"
  if [ "$verdict" != PASS ] && [ "$verdict" != MISMATCH ]; then echo "--- tail $1 ---"; tail -n 15 "/tmp/ua_$1.log"; echo "--- end ---"; fi
}
for i in 1 2 3 4; do
  run "alone-$i" alone
  run "full-$i" ""
done
run model model
echo "SEQUENCE DONE"
