from .benchmark_testing import do_bench, assert_close, perf_report, Benchmark, USE_IPEX_OPTION, BENCHMARKING_METHOD  # type: ignore # noqa: F401

if USE_IPEX_OPTION or BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    from triton.runtime import driver
    from . import benchmark_driver
    # replace the launcher with the profilier hook.
    driver.active.launcher_cls = benchmark_driver.XPULauncher
