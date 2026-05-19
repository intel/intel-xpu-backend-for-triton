#!/bin/bash
# Build-vs-build A/B wrapper for real-workload benchmarks (GEMM, attention).
#
# Link to plan: /home/jovyan/.claude/plans/functional-weaving-seahorse.md §3.5
#
# Usage: evict_last_real_bench.sh <gemm | attention-subset-fwd> [baseline_ref] [patched_ref]
#   Defaults: baseline_ref=main, patched_ref=etiotto/known-reuse-evict-last

set -euo pipefail

# Source oneAPI environment ONCE at top of wrapper. setvars.sh references
# unset variables, so we temporarily disable `-u` while sourcing it.
set +u
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <gemm | attention-subset-fwd> [baseline_ref] [patched_ref]"
    exit 1
fi

BENCHMARK="$1"
BASELINE_REF="${2:-main}"
PATCHED_REF="${3:-etiotto/known-reuse-evict-last}"

WORKDIR="/tmp/evict_last_wt"
OUTPUT_DIR="/tmp/evict_last_real"

mkdir -p "$OUTPUT_DIR"

echo "=== evict_last real-workload A/B wrapper ==="
echo "Benchmark: $BENCHMARK"
echo "Baseline: $BASELINE_REF"
echo "Patched:  $PATCHED_REF"
echo ""

if [ "$BENCHMARK" != "gemm" ] && [ "$BENCHMARK" != "attention-subset-fwd" ]; then
    echo "ERROR: Unknown benchmark '$BENCHMARK'. Must be 'gemm' or 'attention-subset-fwd'."
    exit 1
fi

for SIDE in baseline patched; do
    if [ "$SIDE" = "baseline" ]; then
        REF="$BASELINE_REF"
    else
        REF="$PATCHED_REF"
    fi

    echo "=== Building $SIDE ($REF) ==="
    SIDE_WORKDIR="$WORKDIR/$SIDE"

    # Create or reuse worktree
    if [ -d "$SIDE_WORKDIR" ]; then
        echo "Worktree $SIDE_WORKDIR already exists, reusing."
        cd "$SIDE_WORKDIR"
        git fetch
        git checkout "$REF"
        git pull --ff-only || true
    else
        echo "Creating worktree at $SIDE_WORKDIR for $REF"
        git worktree add "$SIDE_WORKDIR" "$REF"
        cd "$SIDE_WORKDIR"
    fi

    # Build
    echo "Building..."
    PATH="/home/jovyan/.conda/bin:$PATH" MAX_JOBS=12 /bin/bash scripts/compile-triton.sh

    # Run benchmark 5 times
    for RUN in {1..5}; do
        echo ""
        echo "=== Running $SIDE: $BENCHMARK (run $RUN/5) ==="
        RUN_DIR="$OUTPUT_DIR/$SIDE/run_$RUN"
        CACHE_DIR="$RUN_DIR/triton_cache"
        DUMP_DIR="$RUN_DIR/dump"
        mkdir -p "$RUN_DIR" "$CACHE_DIR" "$DUMP_DIR"

        # Re-source oneAPI env immediately before invocation. The top-of-
        # script source can be lost if compile-triton.sh's pip install
        # subprocess resets LD_LIBRARY_PATH inheritance. libsycl.so.8 lives
        # in /opt/intel/oneapi/compiler/2025.3/lib/. Note: setvars.sh has an
        # idempotence guard via SETVARS_COMPLETED — we unset it to force a
        # fresh export.
        set +u
        unset SETVARS_COMPLETED
        source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
        set -u

        # Both benchmarks are env-var driven (no argparse). Device defaults to xpu.
        # gemm_benchmark.py honors TRANSPOSE_A / TRANSPOSE_B env vars.
        # flash_attention_benchmark.py honors FA_KERNEL_MODE (default 'fwd').
        # AnnotateCacheControl is opt-in; enable it on both sides so the A/B
        # measures Phase 2 promotion delta vs Phase 1 (cache-control-on baseline).
        if [ "$BENCHMARK" = "gemm" ]; then
            TRITON_CACHE_DIR="$CACHE_DIR" \
            TRITON_KERNEL_DUMP=1 \
            TRITON_DUMP_DIR="$DUMP_DIR" \
            TRITON_INTEL_DISABLE_ANNOTATE_CACHE_CONTROL=0 \
            /home/jovyan/.conda/bin/python \
                benchmarks/triton_kernels_benchmark/gemm_benchmark.py \
                > "$RUN_DIR/stdout.txt" 2>&1

            if [ -d benchmark_results ]; then
                LATEST_CSV=$(ls -t benchmark_results/*.csv 2>/dev/null | head -1 || echo "")
                if [ -n "$LATEST_CSV" ]; then
                    cp "$LATEST_CSV" "$RUN_DIR/results.csv"
                fi
            fi
        else
            # Forward-only via FA_KERNEL_MODE=fwd (verified at flash_attention_benchmark.py:694).
            TRITON_CACHE_DIR="$CACHE_DIR" \
            TRITON_KERNEL_DUMP=1 \
            TRITON_DUMP_DIR="$DUMP_DIR" \
            TRITON_INTEL_DISABLE_ANNOTATE_CACHE_CONTROL=0 \
            FA_KERNEL_MODE=fwd \
            /home/jovyan/.conda/bin/python \
                benchmarks/triton_kernels_benchmark/flash_attention_benchmark.py \
                > "$RUN_DIR/stdout.txt" 2>&1

            if [ -d benchmark_results ]; then
                LATEST_CSV=$(ls -t benchmark_results/*.csv 2>/dev/null | head -1 || echo "")
                if [ -n "$LATEST_CSV" ]; then
                    cp "$LATEST_CSV" "$RUN_DIR/results.csv"
                fi
            fi
        fi

        # Per-shape TTGIR scan: count evict_last per kernel
        echo "  Scanning TTGIR for evict_last annotations..."
        find "$DUMP_DIR" -name "*.ttgir" | while read -r ttgir; do
            COUNT=$(grep -c "evict_last" "$ttgir" 2>/dev/null || echo "0")
            if [ "$COUNT" -gt 0 ]; then
                KERNEL_NAME=$(basename "$ttgir" .ttgir)
                echo "    $KERNEL_NAME: $COUNT evict_last"
            fi
        done > "$RUN_DIR/evict_last_scan.txt"
    done

    cd - > /dev/null
done

echo ""
echo "=== Analysis: Candidate-friendly shapes ==="
echo "Shapes where patched build has >= 1 evict_last:"

# Scan patched build dumps for candidate shapes
CANDIDATE_SHAPES=()
for RUN in {1..5}; do
    DUMP_DIR="$OUTPUT_DIR/patched/run_$RUN/dump"
    if [ -d "$DUMP_DIR" ]; then
        find "$DUMP_DIR" -name "*.ttgir" | while read -r ttgir; do
            COUNT=$(grep -c "evict_last" "$ttgir" 2>/dev/null || echo "0")
            if [ "$COUNT" -gt 0 ]; then
                KERNEL_NAME=$(basename "$ttgir" .ttgir)
                echo "  $KERNEL_NAME"
            fi
        done | sort -u
        break  # Only need to check one run to identify candidate shapes
    fi
done

echo ""
echo "=== Summary computation ==="
echo "Combining results from 5 runs per side..."

/home/jovyan/.conda/bin/python -c "
import csv
import sys
from pathlib import Path
from collections import defaultdict
import math

output_dir = Path('$OUTPUT_DIR')

# Collect per-shape results from all 5 runs per side
def load_results(side):
    shape_runs = defaultdict(list)  # shape -> [mean1, mean2, ...]
    for run in range(1, 6):
        csv_path = output_dir / side / f'run_{run}' / 'results.csv'
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Expect columns: shape, mean_tflops (or similar)
                # Actual schema depends on gemm_benchmark.py / flash_attention_benchmark.py
                # For robustness, try to find a 'mean' or 'tflops' column
                shape = row.get('shape', row.get('kernel', 'unknown'))
                # Try to extract mean TFLOPS or mean time
                mean_val = None
                for key in ['mean_tflops', 'mean', 'tflops', 'mean_time_ms']:
                    if key in row:
                        try:
                            mean_val = float(row[key])
                            break
                        except ValueError:
                            pass
                if mean_val is not None:
                    shape_runs[shape].append(mean_val)
    return shape_runs

baseline_results = load_results('baseline')
patched_results = load_results('patched')

# Compute mean-of-means and stdev-of-means per shape
def compute_stats(runs):
    if not runs:
        return None, None
    mean_of_means = sum(runs) / len(runs)
    if len(runs) > 1:
        stdev = math.sqrt(sum((x - mean_of_means) ** 2 for x in runs) / (len(runs) - 1))
    else:
        stdev = 0.0
    return mean_of_means, stdev

# Full-suite summary
all_shapes = set(baseline_results.keys()) & set(patched_results.keys())
print(f'Full-suite shapes: {len(all_shapes)}')

ratios = []
for shape in all_shapes:
    b_mean, b_stdev = compute_stats(baseline_results[shape])
    p_mean, p_stdev = compute_stats(patched_results[shape])
    if b_mean and p_mean and b_mean > 0:
        ratio = p_mean / b_mean
        ratios.append(ratio)
        # Check for single-shape regression beyond 3*stdev
        noise_band = 3 * max(b_stdev or 0, p_stdev or 0)
        delta = p_mean - b_mean
        if abs(delta) > noise_band:
            if delta < 0:
                print(f'WARNING: Shape {shape} regresses by {100*(1-ratio):.2f}% (beyond noise band)')

if ratios:
    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f'Full-suite geomean(patched/baseline): {geomean:.4f}')
else:
    print('No overlapping shapes for full-suite geomean')

# Candidate-friendly subset: filter to shapes with evict_last in patched TTGIR
# For simplicity, assume we list candidate shapes in evict_last_scan.txt
# A more robust implementation would parse shape names from TTGIR filenames
candidate_shapes = set()
for run in range(1, 6):
    scan_file = output_dir / 'patched' / f'run_{run}' / 'evict_last_scan.txt'
    if scan_file.exists():
        with open(scan_file) as f:
            for line in f:
                # Format: '  kernel_name: N evict_last'
                parts = line.strip().split(':')
                if parts:
                    candidate_shapes.add(parts[0].strip())

print(f'\\nCandidate-friendly shapes: {len(candidate_shapes)}')

candidate_ratios = []
for shape in candidate_shapes:
    if shape in baseline_results and shape in patched_results:
        b_mean, _ = compute_stats(baseline_results[shape])
        p_mean, _ = compute_stats(patched_results[shape])
        if b_mean and p_mean and b_mean > 0:
            candidate_ratios.append(p_mean / b_mean)

if candidate_ratios:
    candidate_geomean = math.exp(sum(math.log(r) for r in candidate_ratios) / len(candidate_ratios))
    print(f'Candidate-friendly subset geomean(patched/baseline): {candidate_geomean:.4f}')
else:
    print('No candidate-friendly shapes found or no overlapping results')

print(f'\\nPer §3.5 acceptance:')
print(f'  Full-suite: geomean >= 1.00 - noise_band AND no single-shape regression > 3*stdev')
print(f'  Candidate-friendly: geomean shows positive trend exceeding noise band')
"

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Per plan §3.5:"
echo "  - GEMM is merge-gating: full-suite non-regression + candidate-friendly positive trend required."
echo "  - Attention is advisory: reproducible regression beyond noise band is a Phase 2 bug (investigate feedsDotOperand)."
