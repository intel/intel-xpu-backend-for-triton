import os

from .benchmark_testing import (
    assert_close,
    do_bench,
    perf_report,
    Benchmark,
    BENCHMARKING_METHOD,
    filter_providers,
)

if BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    os.environ["INJECT_PYTORCH"] = "True"

__all__ = ["assert_close", "do_bench", "perf_report", "Benchmark", "BENCHMARKING_METHOD", "filter_providers"]
