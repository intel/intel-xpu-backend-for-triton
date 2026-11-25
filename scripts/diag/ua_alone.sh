#!/bin/bash
# EXPERIMENT (#8742, not for merge): soak test. Runs the REAL unified attention benchmark as a stream of fresh
# processes (full config set), stops at the first MISMATCH, and in that same (bad) job immediately runs the
# failing config ALONE in fresh processes plus the model subset. Prints a verdict per process and telemetry.
set +e
N_FULL="${UA_SOAK_N:-12}"
source "$GITHUB_WORKSPACE/scripts/capture-hw-details.sh" > /dev/null 2>&1   # needed by the CLI's report step
cd "$GITHUB_WORKSPACE/vllm" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
state() {  # cheap telemetry to compare good vs bad jobs
  echo "STATE $1: $(uptime | sed 's/.*load average/load average/') | free-mem-MB $(free -m | awk '/Mem:/{print $4"/"$2}')"
  python -c "import torch; f,t=torch.xpu.mem_get_info(); print('STATE $1: xpu free/total GiB %.2f/%.2f'%(f/2**30,t/2**30))" 2>&1 | tail -1
}
VERDICT=""
run() {  # label, UA_ONLY value
  state "before-$1"
  local t0=$SECONDS
  UA_ONLY="$2" TD_PATCHED=0 triton-benchmarks run vllm-unified-attention --reports "$REPORTS" --tag "$1" > "/tmp/ua_$1.log" 2>&1
  local rc=$?
  local sig
  sig=$(grep -oE 'Mismatched elements: [0-9]+ / [0-9]+' "/tmp/ua_$1.log" | head -1)
  if [ -n "$sig" ]; then VERDICT=MISMATCH; elif [ $rc -eq 0 ]; then VERDICT=PASS; else VERDICT="OTHER-FAIL(exit $rc)"; fi
  echo "RESULT $1 (UA_ONLY='$2'): $VERDICT | ${sig:--} | $((SECONDS - t0))s"
  if [ "$VERDICT" != PASS ] && [ "$VERDICT" != MISMATCH ]; then echo "--- tail $1 ---"; tail -n 15 "/tmp/ua_$1.log"; echo "--- end ---"; fi
}
for i in $(seq 1 "$N_FULL"); do
  run "full-$i" ""
  if [ "$VERDICT" = MISMATCH ]; then
    echo "SOAK: first MISMATCH at full-$i; testing the failing config alone in this job"
    for k in 1 2 3; do run "alone-after-fail-$k" alone; done
    run "model-after-fail" model
    run "full-after-fail" ""
    break
  fi
done
echo "SOAK SUMMARY: stopped_after=$i verdict_last=$VERDICT"
echo "SEQUENCE DONE"
