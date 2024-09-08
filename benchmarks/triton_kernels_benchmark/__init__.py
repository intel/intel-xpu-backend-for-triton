import os

from .benchmark_testing import do_bench, assert_close, perf_report, Benchmark  # type: ignore # noqa: F401

if os.getenv("USE_IPEX", "1") == "1":
    from triton.runtime import driver
    from . import benchmark_driver
    # replace the launcher with the profilier hook.
    driver.active.launcher_cls = benchmark_driver.XPULauncher
