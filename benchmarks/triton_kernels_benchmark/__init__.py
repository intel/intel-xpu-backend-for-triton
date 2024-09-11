from .benchmark_testing import do_bench, assert_close, perf_report, Benchmark, USE_IPEX_OPTION  # type: ignore # noqa: F401

if USE_IPEX_OPTION:
    from triton.runtime import driver
    from . import benchmark_driver
    # replace the launcher with the profilier hook.
    driver.active.launcher_cls = benchmark_driver.XPULauncher
