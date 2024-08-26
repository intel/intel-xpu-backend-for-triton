import triton.runtime.driver as driver
from . import benchmark_driver
from .benchmark_testing import do_bench, assert_close, perf_report, Benchmark  # type: ignore # noqa: F401

# replace the launcher with the profilier hook.
driver.active.launcher_cls = benchmark_driver.XPULauncher
