#!/bin/bash
# Build-vs-build A/B wrapper for evict_last microbenchmark.
#
# Link to plan: /home/jovyan/.claude/plans/functional-weaving-seahorse.md §3.2
#
# Usage: evict_last_microbench.sh [baseline_ref] [patched_ref]
#   Defaults: baseline_ref=main, patched_ref=etiotto/known-reuse-evict-last

set -euo pipefail

BASELINE_REF="${1:-main}"
PATCHED_REF="${2:-etiotto/known-reuse-evict-last}"

WORKDIR="/tmp/evict_last_wt"
OUTPUT_DIR="/tmp/evict_last_micro"
CSV_PATH="$OUTPUT_DIR/results.csv"

mkdir -p "$OUTPUT_DIR"
rm -f "$CSV_PATH"

echo "=== evict_last microbenchmark A/B wrapper ==="
echo "Baseline: $BASELINE_REF"
echo "Patched:  $PATCHED_REF"
echo ""

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

    # Run each shape. Per plan §3.2, the 5 repeat-runs MUST be independent
    # subprocesses (fresh JIT state, fresh autotune cache). We invoke the
    # python driver 5 times with --repeat-runs 1 each.
    for SHAPE in primary streaming budget-overflow n-sweep; do
        for RUN in 1 2 3 4 5; do
            echo ""
            echo "=== Running $SIDE: $SHAPE (run $RUN/5) ==="
            CACHE_DIR="$OUTPUT_DIR/$SIDE/$SHAPE/run_$RUN/triton_cache"
            DUMP_DIR="$OUTPUT_DIR/$SIDE/$SHAPE/run_$RUN/dump"
            mkdir -p "$CACHE_DIR" "$DUMP_DIR"

            TRITON_CACHE_DIR="$CACHE_DIR" \
            TRITON_KERNEL_DUMP=1 \
            TRITON_DUMP_DIR="$DUMP_DIR" \
            /home/jovyan/.conda/bin/python \
                benchmarks/triton_kernels_benchmark/evict_last_microbench.py \
                --shape "$SHAPE" \
                --device xpu \
                --repeat-runs 1 \
                --build-label "$SIDE" \
                --csv "$CSV_PATH"
        done
    done

    # IR validation checks per plan §3.3 (use run_1 dump as the canonical sample;
    # IR is deterministic across runs, so any one run is representative).
    echo ""
    echo "=== IR validation for $SIDE (using run_1 dump) ==="
    for SHAPE in primary streaming n-sweep; do
        DUMP_DIR="$OUTPUT_DIR/$SIDE/$SHAPE/run_1/dump"
        COUNT=$(find "$DUMP_DIR" -name "*.ttgir" -exec grep -l "evict_last" {} \; 2>/dev/null | wc -l)
        if [ "$SIDE" = "patched" ] && [ "$COUNT" -eq 0 ]; then
            echo "*** IR check FAILED: $SHAPE has 0 evict_last in patched build (expected >= 1) ***"
        fi
        echo "  $SHAPE: ttgir files containing evict_last = $COUNT"
    done

    # Budget-overflow: expect exactly 1 evict_last (one of two K-loop loads promoted).
    # Count occurrences in TTGIR per kernel file rather than files.
    SHAPE="budget-overflow"
    DUMP_DIR="$OUTPUT_DIR/$SIDE/$SHAPE/run_1/dump"
    COUNT=$(find "$DUMP_DIR" -name "*.ttgir" -exec grep -c "evict_last" {} \; 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ "$SIDE" = "patched" ] && [ "$COUNT" -ne 1 ]; then
        echo "*** IR check FAILED: $SHAPE has $COUNT evict_last occurrences in patched build (expected exactly 1) ***"
    fi
    echo "  $SHAPE: evict_last occurrences = $COUNT"

    cd - > /dev/null
done

echo ""
echo "=== Delta summary (patched vs baseline) ==="
echo "Reading results from $CSV_PATH"

/home/jovyan/.conda/bin/python -c "
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

csv_path = Path('$CSV_PATH')
if not csv_path.exists():
    print('ERROR: CSV not found')
    sys.exit(1)

# Each (side, shape, M, N, K) has up to 5 rows (one per subprocess run).
# Aggregate them: collect per-run mean_tflops, then compute mean-of-means
# and stdev-of-means per plan section 3.2.
buckets = defaultdict(list)  # (label, shape, M, N, K) -> [tflops_run1, ...]
with open(csv_path) as f:
    for row in csv.DictReader(f):
        key = (row['build_label'], row['shape_name'], int(row['M']), int(row['N']), int(row['K']))
        buckets[key].append(float(row['mean_tflops']))

def stats(samples):
    n = len(samples)
    if n == 0:
        return None, None
    m = sum(samples) / n
    s = math.sqrt(sum((x - m) ** 2 for x in samples) / (n - 1)) if n > 1 else 0.0
    return m, s

# Pair baseline vs patched
shape_keys = sorted({(s, M, N, K) for (_, s, M, N, K) in buckets})

header = f'{\"Shape\":<18} {\"M\":>5} {\"N\":>5} {\"K\":>5} {\"Baseline TFLOPS\":>18} {\"Patched TFLOPS\":>18} {\"Delta %\":>9} {\"Actionable\":>11}'
print(header)
print('-' * len(header))

for shape, M, N, K in shape_keys:
    b_mean, b_stdev = stats(buckets.get(('baseline', shape, M, N, K), []))
    p_mean, p_stdev = stats(buckets.get(('patched', shape, M, N, K), []))
    if b_mean is None or p_mean is None:
        continue
    delta_pct = 100 * (p_mean - b_mean) / b_mean if b_mean > 0 else 0.0
    noise_pct = 100 * 3 * max(b_stdev, p_stdev) / b_mean if b_mean > 0 else 0.0
    actionable = 'YES' if abs(delta_pct) >= noise_pct else 'NO'
    print(f'{shape:<18} {M:>5} {N:>5} {K:>5} {b_mean:>9.3f} +/- {b_stdev:>5.3f} {p_mean:>9.3f} +/- {p_stdev:>5.3f} {delta_pct:>8.2f}% {actionable:>11}')
"

echo ""
echo "Results saved to: $CSV_PATH"
