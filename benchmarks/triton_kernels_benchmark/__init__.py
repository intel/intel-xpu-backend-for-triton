from .benchmark_testing import do_bench, make_do_bench_for_autotune, assert_close, perf_report, Benchmark, BENCHMARKING_METHOD  # type: ignore # noqa: F401

if BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    from triton.runtime import driver
    from . import benchmark_driver
    # replace the launcher with the profilier hook.
    driver.active.launcher_cls = benchmark_driver.XPULauncher
