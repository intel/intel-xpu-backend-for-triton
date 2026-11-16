#!/bin/bash
# Run repro.py with P concurrent processes per config (split / nosplit / ticket).
# usage: run_matrix.sh <outdir> <iters> <timeout_s> "<cfg> <cfg> ..."   cfg = <split|nosplit>:<P>
#   e.g. run_matrix.sh results/r1 20 600 "split:1 split:2 split:3 nosplit:3"
# Env: VENV (optional, activated if set), ONEAPI_SETVARS (default /opt/intel/oneapi/setvars.sh).
# Per config: results/<cfg>/p<i>.jsonl (iterations), p<i>.err (stderr; faulthandler dump on hang), summary line.
OUT=$1; ITERS=$2; TO=$3; CFGS=$4
HERE=$(cd "$(dirname "$0")" && pwd)
[ -n "${VENV:-}" ] && . "$VENV/bin/activate"
set +u; source "${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}" >/dev/null 2>&1; set -u
mkdir -p "$OUT"; echo "start $(date -u +%FT%TZ) iters=$ITERS timeout=$TO cfgs=$CFGS" | tee -a "$OUT/summary.txt"
for cfg in $CFGS; do
  mode=${cfg%%:*}; P=${cfg##*:}; D=$OUT/$cfg; mkdir -p "$D"; flag=""; [ "$mode" = nosplit ] && flag=--no-split; [ "$mode" = ticket ] && flag=--ticket
  if ! timeout 60 python -c "import torch; x=torch.ones(1024, device='xpu'); torch.xpu.synchronize(); assert x.sum().item()==1024" >/dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) $cfg: GPU HEALTH CHECK FAILED before start -> aborting matrix" | tee -a "$OUT/summary.txt"; break
  fi
  t0=$(date +%s); pids=()
  for i in $(seq 0 $((P-1))); do
    python "$HERE/repro.py" --iters "$ITERS" --timeout "$TO" $flag --tag "$cfg-p$i" > "$D/p$i.jsonl" 2> "$D/p$i.err" & pids+=($!)
  done
  rc=(); for p in "${pids[@]}"; do wait "$p"; rc+=($?); done
  hangs=$(grep -l "Timeout (" "$D"/p*.err 2>/dev/null | wc -l)
  its=$(cat "$D"/p*.jsonl | grep -c '"event": "iter"')
  maxerr=$(cat "$D"/p*.jsonl | grep -o '"rel_err": [0-9.e-]*' | awk '{print $2}' | sort -g | tail -1)
  med=$(cat "$D"/p*.jsonl | grep -o '"sec": [0-9.]*' | awk '{print $2}' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  echo "$(date -u +%FT%TZ) $cfg: procs=$P hung_procs=$hangs iters_ok=$its/$((P*ITERS)) median_iter_s=$med max_rel_err=$maxerr rc=${rc[*]} wall=$(( $(date +%s)-t0 ))s" | tee -a "$OUT/summary.txt"
done
echo "end $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
