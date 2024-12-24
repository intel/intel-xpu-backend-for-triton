import os

from .benchmark_testing import BENCHMARKING_METHOD

if BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    os.environ["INJECT_PYTORCH"] = "True"
