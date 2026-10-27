"""
Pre-warm the Triton disk compile cache for any triton_kernels_benchmark script that uses
the shared benchmark_testing.py perf_report/Mark harness (gemm_*, flash_attention_*,
fused_gemm_benchmark.py, etc), before running the benchmark itself.

Why this is needed:
- `Autotuner._parallel_warmup()` dispatches one shape's configs to the warmup process pool
  at a time and blocks until they finish before that shape's (serial, GPU-bound)
  benchmarking phase runs, so the pool never gets fed enough work at once to use all
  `TRITON_AUTOTUNE_WARMUP_WORKERS` continuously.
- Triton's disk-cache key includes the kernel function's module identity
  (`fn.__module__`). Running `python foo_benchmark.py` directly makes that module
  `"__main__"`; importing the file as a package module gives a different name - same
  source, different cache key. This script loads the target file via
  `runpy.run_path(..., run_name="__main__")` to get the exact identity the real run uses.

Rather than hand-reconstructing each kernel's shapes/strides/grid per benchmark (which
doesn't generalize - every benchmark's kernel signature differs), this monkeypatches
`Autotuner.run` process-wide to capture-and-dispatch every attempted autotuned kernel
launch to the warmup pool instead of compiling+launching it, and lets the target
benchmark's own existing shape/provider sweep (`Mark.run()`) decide which shapes get
visited. `disable_verification()` skips correctness checks (nothing real is computed to
check), and `get_do_bench` is replaced so the benchmarking loop calls its function once
instead of looping `n_warmup + n_repeat` times - not needed for warmup, and it also
eliminates the real GPU timing cost for non-Triton providers (onednn, sycl-tla, ...) in
the same sweep.

Only `@triton.autotune`-wrapped kernels get pool-parallelized this way (that's where the
16-configs-per-shape serialization problem lives). A plain `@triton.jit` kernel called
without `@triton.autotune` still compiles for real, in-process, the first time each shape
is seen - not patched, since there's no autotuner config list to parallelize across and
patching `JITFunction.run` directly would deadlock `dispatcher.dispatch()`'s own internal
use of it (see `patched_autotuner_run`'s comment).

Run this once with the real worker count, then run the benchmark normally - its
autotuning will hit the now-warm disk cache instead of compiling inline.

    TRITON_AUTOTUNE_WARMUP_WORKERS=19 python warmup_benchmark_cache.py gemm_tensor_of_ptr_benchmark.py
    python gemm_tensor_of_ptr_benchmark.py

Caveat: this has been validated against gemm_tensor_of_ptr_benchmark.py and
gemm_splitk_benchmark.py. Benchmarks with unusual structure (multi-kernel pipelines,
non-`@triton.autotune` kernels with expensive per-call Python-side setup, results that
feed back into later shapes) may need a check that the full sweep actually ran - the
printed capture count is a sanity signal, but a benchmark-specific bug could still under-
or over-count.
"""
import atexit
import copy
import os
import runpy
import sys

import triton
import triton_kernels_benchmark
from triton.runtime.autotuner import Autotuner, _get_warmup_dispatcher
from triton._compile_warmup_pool import _UnsupportedPreloadError
from triton_kernels_benchmark import benchmark_testing


def _fake_do_bench(fn, *_args, quantiles=None, **_kwargs):
    fn()
    if quantiles is not None:
        return [1.0] * (len(quantiles) + 2)
    return 1.0


def _fake_get_do_bench(n_warmup, n_repeat, quantiles):  # pylint: disable=unused-argument
    return lambda fn, **kwargs: _fake_do_bench(fn, quantiles=kwargs.get('quantiles', quantiles))


def _install_capture_hooks(workers: int):
    dispatcher = _get_warmup_dispatcher(workers)
    captured = {"count": 0}
    original_autotuner_run = Autotuner.run

    def patched_autotuner_run(self, *args, grid, warmup, **kwargs):
        if warmup:
            return original_autotuner_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        self.nargs = dict(zip(self.arg_names, args))
        try:
            configs = self.prune_configs(kwargs)
        finally:
            self.nargs = None
        kernel = copy.copy(self)
        kernel.prune_configs = lambda kwargs, configs=configs: configs
        rest = {k: v for k, v in kwargs.items() if k not in ('grid', 'warmup')}
        # `dispatcher.dispatch()` itself calls `kernel.fn.warmup()` (plain JITFunction.run,
        # warmup=True) to capture the specialization for the pool - that must hit the real
        # JITFunction.run, not this patch, so only Autotuner.run is intercepted here.
        dispatcher.dispatch(*args, kernel=kernel, grid=grid, test=None, **rest)
        captured["count"] += len(configs)
        return None

    Autotuner.run = patched_autotuner_run

    def restore():
        Autotuner.run = original_autotuner_run

    return dispatcher, captured, restore


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <benchmark_script.py>")
    script_path = os.path.abspath(sys.argv[1])
    # The target script parses its own CLI flags from sys.argv (e.g. --reports, --n_runs);
    # strip our own argument so it doesn't see it as an unexpected positional.
    sys.argv = [script_path]

    workers = triton.knobs.autotuning.warmup_workers
    if workers <= 0:
        raise SystemExit("Set TRITON_AUTOTUNE_WARMUP_WORKERS to a positive worker count before running this script.")

    benchmark_testing.disable_verification()
    original_get_do_bench = triton_kernels_benchmark.get_do_bench
    triton_kernels_benchmark.get_do_bench = _fake_get_do_bench
    benchmark_testing.get_do_bench = _fake_get_do_bench

    dispatcher, captured, restore_run_patches = _install_capture_hooks(workers)
    try:
        runpy.run_path(script_path, run_name="__main__")
    finally:
        restore_run_patches()
        triton_kernels_benchmark.get_do_bench = original_get_do_bench
        benchmark_testing.get_do_bench = original_get_do_bench

    print(f"Captured {captured['count']} kernel launches across the benchmark's shape/provider sweep.")

    failures = []
    for future in dispatcher._pending:  # pylint: disable=protected-access
        try:
            future.result()
        except _UnsupportedPreloadError as error:
            dispatcher.tolerate_failure(future)
            failures.append(str(error))

    if failures:
        print(f"WARNING: {len(failures)}/{len(dispatcher._pending)} preloads failed and were NOT cached "
              f"(the disk cache was not warmed for these). First failure:\n{failures[0]}")

    dispatcher.finish()
    # `_get_warmup_dispatcher` registered this same `finish` at atexit; we already
    # drained the pool above, so unregister it to avoid a second, empty metrics line.
    atexit.unregister(dispatcher.finish)


if __name__ == '__main__':
    main()
