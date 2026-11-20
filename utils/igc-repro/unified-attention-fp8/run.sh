#!/usr/bin/env bash
# Build and run the standalone reproducer(s) in cases/*. Needs: g++, libze_loader (+ Level Zero headers), ocloc.
# Writes results/summary.txt, results/*.bin (zebins for diffing across IGC versions) and results/run.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ITERS="${ITERATIONS:-50}"; WARMUP="${WARMUP:-10}"; DEVICE="${OCLOC_DEVICE:-bmg}"
INC=""; for d in /usr/include include; do [ -f "$d/level_zero/ze_api.h" ] && INC="-I$d" && break; done
[ -n "$INC" ] || { echo "Level Zero headers (level_zero/ze_api.h) not found; install level-zero-dev or drop them in ./include" >&2; exit 1; }
g++ -O2 -std=c++17 repro.cpp $INC -o repro -lze_loader
mkdir -p results; S=results/summary.txt
{
  echo "date: $(date -u +%FT%TZ)  host: $(hostname)"
  echo "libigc: $(dpkg-query --show --showformat='${Version}' libigc2 2>/dev/null || dpkg-query --show --showformat='${Version}' libigc1 2>/dev/null || echo unknown)"
  echo "compute-runtime: $(dpkg-query --show --showformat='${Version}' intel-opencl-icd 2>/dev/null || echo unknown)"
  echo "ocloc: $(ocloc --version 2>&1 | head -1)"
  echo
} > "$S"
VARIANTS=("default::" "grf256::-cl-intel-256-GRF-per-thread")
for c in cases/*/; do
  name=$(basename "$c")
  for v in "${VARIANTS[@]}"; do
    vn="${v%%::*}"; vf="${v#*::}"
    tmp=$(mktemp -d); ocloc compile -file "$c/kernel.spv" -spirv_input -device "$DEVICE" -options "$vf" -output "$tmp/k" > "$tmp/log" 2>&1 || { cat "$tmp/log"; echo "$name/$vn: ocloc FAILED" >> "$S"; continue; }
    cp "$tmp"/k_*.bin "results/${name}_${vn}.bin"
    spill=$(grep -a -o 'spill_size: *[0-9]*' "results/${name}_${vn}.bin" | head -1 | grep -o '[0-9]*$' || echo '?')
    grf=$(grep -a -o 'grf_count: *[0-9]*' "results/${name}_${vn}.bin" | head -1 | grep -o '[0-9]*$' || echo '?')
    res=$(./repro "$c/case.cfg" --build-flags "$vf" --iters "$ITERS" --warmup "$WARMUP" | grep '^RESULT')
    echo "$name  variant=$vn  ocloc: grf=$grf spill=${spill}B  |  ${res#RESULT }" | tee -a "$S"
    rm -rf "$tmp"
  done
done
echo; cat "$S"
