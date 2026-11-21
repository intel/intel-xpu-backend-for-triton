#!/bin/bash
# EXPERIMENT (#8742, not for merge): does the bad process state occur per JOB or per PROCESS, and does the failing
# config ever fail when run ALONE in a fresh process during a job where the FULL run fails?
# Interleaves alone/full processes so one job gives several independent samples of each.
set +e
source "$GITHUB_WORKSPACE/scripts/capture-hw-details.sh" > /dev/null 2>&1   # needed by the CLI's report step
cd "$GITHUB_WORKSPACE/vllm" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
state() {  # cheap telemetry to compare good vs bad jobs (concurrent load, memory)
  echo "STATE $1: $(uptime | sed 's/.*load average/load average/') | free-mem-MB $(free -m | awk '/Mem:/{print $4"/"$2}')"
  python -c "import torch; f,t=torch.xpu.mem_get_info(); print('STATE $1: xpu free/total GiB %.2f/%.2f'%(f/2**30,t/2**30))" 2>&1 | tail -1
  ps -eo pid,etimes,pcpu,comm --sort=-pcpu | head -6 | sed 's/^/STATE ps: /'
}
run() {  # label, UA_ONLY value
  state "before-$1"
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
