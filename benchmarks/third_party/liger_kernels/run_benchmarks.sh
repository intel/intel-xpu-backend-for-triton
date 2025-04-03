#!/usr/bin/env bash

set -euo pipefail


# Array to keep track of failed benchmarks
FAILED_BENCHMARKS=()

for file in Liger-Kernel/benchmark/scripts/benchmark_*; do
    if python "$file"; then
        echo "Benchmark ran successfully: $file"
    else
        echo "Error: Benchmark failed for $file."
        FAILED_BENCHMARKS+=("$file")
    fi
done

# Print failed benchmarks
if [ ${#FAILED_BENCHMARKS[@]} -ne 0 ]; then
    echo "The following benchmarks failed:"
    for failed_bench in "${FAILED_BENCHMARKS[@]}"; do
        echo "$failed_bench"
    done
    exit 1
else
    echo "All benchmarks completed successfully."
fi

exit 0
