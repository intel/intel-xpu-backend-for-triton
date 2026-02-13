import os

from triton_kernels_benchmark.benchmark_testing import (
    assert_close,
    do_bench,
    filter_providers,
    perf_report,
    Benchmark,
    BenchmarkCategory,
    BenchmarkConfig,
    BENCHMARKING_METHOD,
    get_do_bench,
)

from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser

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
    "BENCHMARKING_METHOD",
    "ShapePatternParser",
    "get_do_bench",
]
