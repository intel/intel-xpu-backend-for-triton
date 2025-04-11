#!/usr/bin/env bash

set -euo pipefail


# Array to keep track of failed benchmarks
FAILED_BENCHMARKS=()

for file in Liger-Kernel/benchmark/scripts/benchmark_*; do
    # TODO: unskip when https://github.com/intel/intel-xpu-backend-for-triton/issues/3873 is resolved
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_tvd.py" ]; then
        continue
    fi
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_orpo_loss.py" ]; then
        continue
    fi
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_qwen2vl_mrope.py" ]; then
        continue
    fi
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
