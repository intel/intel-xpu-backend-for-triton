#!/usr/bin/env bash

set -euo pipefail

for file in Liger-Kernel/benchmark/scripts/benchmark_*; do
    python "$file"
done
