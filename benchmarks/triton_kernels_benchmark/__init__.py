import torch
import intel_extension_for_pytorch
from . import xetla_kernel
from . import benchmark_testing
from .benchmark_testing import do_bench, assert_close, perf_report, Benchmark

import triton
import triton.runtime.driver as driver
from . import benchmark_driver

# replace the launcher with the profilier hook.
driver.active.launcher_cls = benchmark_driver.XPULauncher
