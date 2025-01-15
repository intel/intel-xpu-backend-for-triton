import os

from .benchmark_testing import assert_close, make_do_bench_for_autotune, do_bench, perf_report, Benchmark, BENCHMARKING_METHOD

if BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    os.environ["INJECT_PYTORCH"] = "True"

__all__ = [
    "assert_close",
    "make_do_bench_for_autotune",
    "do_bench",
    "perf_report",
    "Benchmark",
    "BENCHMARKING_METHOD",
]
