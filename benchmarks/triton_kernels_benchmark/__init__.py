import os

from triton.testing import assert_close

from .benchmark_testing import do_bench, perf_report, Benchmark, BENCHMARKING_METHOD

if BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    os.environ["INJECT_PYTORCH"] = "True"

__all__ = [
    "assert_close",
    "do_bench",
    "perf_report",
    "Benchmark",
    "BENCHMARKING_METHOD",
]
