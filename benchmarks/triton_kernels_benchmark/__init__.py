import os

import triton

from triton_kernels_benchmark.benchmark_testing import (
    DEVICE,
    assert_close,
    do_bench,
    filter_providers,
    perf_report,
    Benchmark,
    BenchmarkCategory,
    BenchmarkConfig,
    BENCHMARKING_CONFIG,
    BENCHMARKING_METHOD,
    get_do_bench,
    get_total_gpu_memory_bytes,
)

from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser

# GRF mode for configs that request a large register file: CRI supports 512 GRFs, other targets 256.
LARGE_GRF_MODE = "512" if DEVICE == "xpu" and triton.runtime.driver.active.get_current_target().arch.get(
    "arch") == "cri" else "256"

if BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    os.environ["INJECT_PYTORCH"] = "True"

__all__ = [
    "assert_close",
    "do_bench",
    "filter_providers",
    "perf_report",
    "Benchmark",
    "BenchmarkCategory",
    "BenchmarkConfig",
    "BENCHMARKING_CONFIG",
    "BENCHMARKING_METHOD",
    "ShapePatternParser",
    "get_do_bench",
    "LARGE_GRF_MODE",
]
