"""Kernel launch overhead benchmarks. The kernels are empty, so every result is host time.

``launch-overhead``: cost of one launch by the number and type of the kernel arguments (``arg_type``):

* ``none``: no arguments, the fixed cost of a launch;
* ``ptr``: ``torch.Tensor`` pointers;
* ``i32``: integer scalars (other scalar types cost the same);
* ``td``: 2D ``TensorDescriptor`` arguments.

It is measured at two levels of the launch path (``level``):

* ``jit``: ``kernel[grid](*args)``, the full user-facing call (Python JIT dispatch and the launcher);
* ``launcher``: ``CompiledKernel.run(...)``, the backend launcher alone (argument packing and submission).

``jit - launcher`` is the Python JIT dispatch cost. Metrics:

* ``time_us``: median wall time per launch;
* ``ns_per_arg``: cost of one argument (one descriptor for ``td``) over the empty kernel at the same level;
* ``cpu_mhz``: frequency of the launching CPU during the measurement.

``launch-overhead-cold``: ``time_ms`` of the first launches in a fresh process, after PyTorch has initialized the
device. The Triton and GPU driver compilation caches are empty (``cache=cold``) or populated by an earlier
process (``cache=warm``). Phases (``phase``):

* ``driver``: Triton driver initialization, which builds (cold) or loads (warm) the launcher module;
* ``hash``: ``triton_key()``, the hash of the Triton installation in every cache key, computed once per process;
* ``kernel``: first launch of a kernel: compile it or load it from the cache, load the binary, submit;
* ``total``: ``driver + hash + kernel``, the delay of the first launch in the process;
* ``nextkernel``: first launch of a second kernel, the per-kernel part of ``total``.

Each measurement runs in a fresh worker process, so that results do not depend on benchmarks run earlier in the
same process. It also measures the production launcher: importing the benchmark suite sets ``INJECT_PYTORCH``,
which instruments every launch for the PyTorch profiler. The worker pins its launching thread to one CPU, because
launch cost follows the CPU frequency. Set ``LAUNCH_OVERHEAD_CPU`` to an otherwise idle core on the GPU's NUMA node
(default: the second CPU of the affinity mask, ``-1`` disables pinning) and keep a fixed frequency policy (e.g. the
``performance`` governor).

Usage: ``triton-benchmarks run launch-overhead --reports <dir>`` and
``triton-benchmarks run launch-overhead-cold --reports <dir>``.
"""
from __future__ import annotations

import argparse
import json
import linecache
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Union

ARG_COUNTS = {"none": [0], "ptr": [4, 16, 64], "i32": [4, 16, 64], "td": [1, 4]}
LEVELS = ["jit", "launcher"]
# Signature of the kernels launched by the first-launch benchmark.
FIRST_LAUNCH_ARGS = ("ptr", 4)
FIRST_LAUNCH_PHASES = ["total", "driver", "hash", "kernel", "nextkernel"]
CPU_ENV_VAR = "LAUNCH_OVERHEAD_CPU"
WORKER_TIMEOUT_S = 600


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return ""


def cpu_model() -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def report_params() -> Dict[str, str]:
    """Params added to every report row: launch overhead is comparable only on the same CPU model."""
    return {"cpu": cpu_model()}


def _cpufreq(cpu: int, name: str) -> str:
    return _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}")


def _cpu_mhz(cpu: int) -> float:
    khz = _cpufreq(cpu, "scaling_cur_freq")
    return int(khz) / 1000 if khz else math.nan


def launch_cpu() -> int:
    """CPU to pin the launching thread to, or -1 to not pin it."""
    if not hasattr(os, "sched_setaffinity"):
        return -1
    if CPU_ENV_VAR in os.environ:
        return int(os.environ[CPU_ENV_VAR])
    cpus = sorted(os.sched_getaffinity(0))
    # CPU 0 usually serves more interrupts and housekeeping work than the others.
    return cpus[1] if len(cpus) > 1 else cpus[0]


def _empty_kernel(name: str, num_args: int):
    import triton  # pylint: disable=import-outside-toplevel

    params = ", ".join(f"a{i}" for i in range(num_args))
    src = f"def {name}({params}):\n    pass\n"
    filename = f"<{name}>"
    # @triton.jit reads the kernel source with inspect, so register the generated source in linecache.
    linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
    namespace = {}
    exec(compile(src, filename, "exec"), namespace)  # pylint: disable=exec-used
    return triton.jit(namespace[name])


def _make_args(arg_type: str, num_args: int, device) -> list:
    import torch  # pylint: disable=import-outside-toplevel
    from triton.tools.tensor_descriptor import TensorDescriptor  # pylint: disable=import-outside-toplevel

    if arg_type == "none":
        return []
    if arg_type == "ptr":
        return [torch.empty(16, device=device) for _ in range(num_args)]
    if arg_type == "i32":
        # Neither 1 nor a multiple of 16, so the JIT does not specialize the kernel on them.
        return [17 + 16 * i for i in range(num_args)]
    if arg_type == "td":
        return [TensorDescriptor.from_tensor(torch.empty(64, 64, device=device), [16, 16]) for _ in range(num_args)]
    raise ValueError(f"Unknown argument type {arg_type!r}")


def _launch_loop(arg_type: str, num_args: int, level: str, device) -> Callable[[int], None]:
    """Compiles and launches the kernel once, then returns `loop(n)`, which launches it `n` times at `level`."""
    from triton.runtime import driver  # pylint: disable=import-outside-toplevel

    kernel = _empty_kernel(f"launch_overhead_{arg_type}_{num_args}", num_args)
    args = _make_args(arg_type, num_args, device)
    compiled = kernel[(1, )](*args)
    if level == "jit":

        def jit_loop(n: int):
            for _ in range(n):
                kernel[(1, )](*args)

        return jit_loop
    if level == "launcher":
        stream = driver.active.get_current_stream(driver.active.get_current_device())
        run, function, metadata = compiled.run, compiled.function, compiled.packed_metadata

        def launcher_loop(n: int):
            # Same call as in JITFunction.run, without launch metadata and hooks.
            for _ in range(n):
                run(1, 1, 1, stream, function, metadata, None, None, None, *args)

        return launcher_loop
    raise ValueError(f"Unknown level {level!r}")


def _host_info(cpu: int, device_module) -> Dict[str, Union[str, int]]:
    import torch  # pylint: disable=import-outside-toplevel
    import triton  # pylint: disable=import-outside-toplevel

    freq_cpu = max(cpu, 0)
    max_khz = _cpufreq(freq_cpu, "cpuinfo_max_freq")
    return {
        "cpu_model": cpu_model(),
        "pinned_cpu": cpu,
        "governor": _cpufreq(freq_cpu, "scaling_governor"),
        "epp": _cpufreq(freq_cpu, "energy_performance_preference"),
        "max_mhz": int(max_khz) // 1000 if max_khz else "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": device_module.get_device_name(),
        "SYCL_UR_USE_LEVEL_ZERO_V2": os.environ.get("SYCL_UR_USE_LEVEL_ZERO_V2", "<unset>"),
    }


def measure_launch(arg_type: str, num_args: int, n_launches: int, reps: int, warmup_launches: int, cpu: int) -> dict:
    """Worker side: per-repetition time per launch (us) at each level, of the kernel and of the empty kernel."""
    import torch  # pylint: disable=import-outside-toplevel
    from triton.runtime import driver  # pylint: disable=import-outside-toplevel

    device = driver.active.get_active_torch_device()
    device_module = getattr(torch, device.type)
    loops = {}
    for level in LEVELS:
        loops[level, "kernel"] = _launch_loop(arg_type, num_args, level, device)
        if num_args:
            loops[level, "baseline"] = _launch_loop("none", 0, level, device)
    device_module.synchronize()
    if cpu >= 0:
        # Pin only the launching thread; runtime threads started during initialization keep their affinity.
        os.sched_setaffinity(0, {cpu})
    for loop in loops.values():
        loop(warmup_launches)
    samples_us: Dict[str, Dict[str, List[float]]] = {level: {} for level in LEVELS}
    cpu_mhz = []
    for _ in range(reps):
        # Interleave all loops so that they run under the same conditions.
        for (level, name), loop in loops.items():
            device_module.synchronize()
            start = time.perf_counter_ns()
            loop(n_launches)
            device_module.synchronize()
            samples_us[level].setdefault(name, []).append((time.perf_counter_ns() - start) / n_launches / 1e3)
        if cpu >= 0:
            cpu_mhz.append(_cpu_mhz(cpu))
    return {"samples_us": samples_us, "cpu_mhz": cpu_mhz, "host": _host_info(cpu, device_module)}


def measure_first_launch(arg_type: str, num_args: int, cpu: int) -> dict:
    """Worker side: duration (ms) of each phase of the first launches in this process."""
    import torch  # pylint: disable=import-outside-toplevel
    from triton.runtime import driver  # pylint: disable=import-outside-toplevel
    from triton.runtime.cache import triton_key  # pylint: disable=import-outside-toplevel

    # Only PyTorch initializes the device here: the Triton driver initialization is a measured phase.
    device = torch.accelerator.current_accelerator()
    device_module = getattr(torch, device.type)
    args = _make_args(arg_type, num_args, device)
    device_module.synchronize()
    if cpu >= 0:
        os.sched_setaffinity(0, {cpu})
    first_kernel = _empty_kernel(f"first_launch_{arg_type}_{num_args}", num_args)
    next_kernel = _empty_kernel(f"next_launch_{arg_type}_{num_args}", num_args)

    def init_driver():
        driver.active.get_current_target()
        getattr(driver.active, "utils")  # Loads the launcher module.

    def launch(kernel):
        kernel[(1, )](*args)
        device_module.synchronize()

    phases_ms = {}
    for phase, step in (("driver", init_driver), ("hash", triton_key), ("kernel", lambda: launch(first_kernel)),
                        ("nextkernel", lambda: launch(next_kernel))):
        start = time.perf_counter_ns()
        step()
        phases_ms[phase] = (time.perf_counter_ns() - start) / 1e6
    phases_ms["total"] = phases_ms["driver"] + phases_ms["hash"] + phases_ms["kernel"]
    return {"phases_ms": phases_ms, "host": _host_info(cpu, device_module)}


def _run_worker(mode: str, options: Dict[str, Union[str, int]], env: Optional[Dict[str, str]] = None) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", mode]
    for name, value in options.items():
        cmd += [f"--{name}", str(value)]
    # The benchmark suite sets INJECT_PYTORCH, which instruments every launch; measure the production launcher.
    worker_env = {name: value for name, value in os.environ.items() if name != "INJECT_PYTORCH"}
    worker_env.update(env or {})
    result = subprocess.run(cmd, env=worker_env, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Launch overhead worker failed: {shlex.join(cmd)}\n{result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _first_launch_samples(cold_reps: int, warm_reps: int) -> Dict[str, List[dict]]:
    """Runs workers with empty caches, then with the caches populated by the last of them."""
    if cold_reps < 1:
        raise ValueError("The first-launch benchmark needs at least one cold sample to populate the caches")
    arg_type, num_args = FIRST_LAUNCH_ARGS
    options = {"arg-type": arg_type, "num-args": num_args, "cpu": launch_cpu()}
    with tempfile.TemporaryDirectory(prefix="launch-overhead-") as tmpdir:
        # A fresh Triton cache and Intel GPU driver (NEO) compiler cache for every cold sample.
        cache_envs = []
        for i in range(cold_reps):
            triton_dir = os.path.join(tmpdir, str(i), "triton")
            neo_dir = os.path.join(tmpdir, str(i), "neo")
            # NEO disables the persistent cache if NEO_CACHE_DIR does not exist.
            os.makedirs(triton_dir, exist_ok=True)
            os.makedirs(neo_dir, exist_ok=True)
            cache_envs.append({
                "TRITON_CACHE_DIR": triton_dir,
                "NEO_CACHE_PERSISTENT": "1",
                "NEO_CACHE_DIR": neo_dir,
            })
        samples = {"cold": [_run_worker("first-launch", options, env) for env in cache_envs]}
        samples["warm"] = [_run_worker("first-launch", options, cache_envs[-1]) for _ in range(warm_reps)]
    return samples


class _SharedMeasurement:
    """Shares one measurement between several benchmark points.

    Each point gets the result once; asking for it again (the next of `n_runs` runs) measures again.
    """

    def __init__(self, measure: Callable[..., dict]):
        self._measure = measure
        self._results: Dict[tuple, dict] = {}
        self._served: Dict[tuple, set] = {}

    def get(self, key: tuple, point) -> dict:
        if key not in self._results or point in self._served[key]:
            self._results[key] = self._measure(*key)
            self._served[key] = set()
        self._served[key].add(point)
        return self._results[key]


_printed_host_info = False


def _print_host_info_once(host: Dict[str, Union[str, int]]):
    global _printed_host_info  # pylint: disable=global-statement
    if _printed_host_info:
        return
    _printed_host_info = True
    print("Launch overhead host: " + ", ".join(f"{name}={value}" for name, value in host.items()))
    governor = host["governor"]
    if governor not in ("performance", ""):
        print(f"Warning: CPU frequency governor is {governor!r}, launch overhead results may be unstable.")


def _cv(values: List[float]) -> float:
    return statistics.stdev(values) / statistics.fmean(values) if len(values) > 1 else 0.0


def get_benchmark(providers_filter: Optional[List[str]] = None, n_launches: int = 5000, reps: int = 11,
                  warmup_launches: int = 500):
    """Returns a Mark object that runs the launch overhead sweep."""
    import triton_kernels_benchmark as benchmark_suite  # pylint: disable=import-outside-toplevel

    # Long reports need the DB compiler name as the provider label.
    supported_providers = {"triton": "triton"}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    # One worker measures all levels of a signature.
    measurements = _SharedMeasurement(lambda arg_type, num_args: _run_worker(
        "launch", {
            "arg-type": arg_type, "num-args": num_args, "n-launches": n_launches, "reps": reps, "warmup-launches":
            warmup_launches, "cpu": launch_cpu()
        }))

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["arg_type", "num_args", "level"],
            x_vals=[(arg_type, num_args, level)
                    for arg_type, counts in ARG_COUNTS.items()
                    for num_args in counts
                    for level in LEVELS],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            ylabel=["time_us", "ns_per_arg", "cpu_mhz"],
            plot_name="launch-overhead-sweep",
            args={},
        ))
    def benchmark(arg_type, num_args, level, provider):
        if provider != "triton":
            raise NotImplementedError(f"Unsupported provider {provider}")
        result = measurements.get((arg_type, num_args), level)
        _print_host_info_once(result["host"])
        samples = result["samples_us"][level]
        times = samples["kernel"]
        time_us = statistics.median(times)
        ns_per_arg = math.nan
        if num_args:
            ns_per_arg = (time_us - statistics.median(samples["baseline"])) * 1e3 / num_args
        freqs = [freq for freq in result["cpu_mhz"] if not math.isnan(freq)]
        cpu_mhz = statistics.median(freqs) if freqs else math.nan
        return (time_us, min(times), max(times)), ns_per_arg, cpu_mhz, _cv(times)

    return benchmark


def get_first_launch_benchmark(providers_filter: Optional[List[str]] = None, cold_reps: int = 3, warm_reps: int = 5):
    """Returns a Mark object that measures the first launches in a fresh process."""
    import triton_kernels_benchmark as benchmark_suite  # pylint: disable=import-outside-toplevel

    supported_providers = {"triton": "triton"}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    # The warm samples reuse the caches of the cold ones, so all points share one set of workers.
    measurements = _SharedMeasurement(lambda: _first_launch_samples(cold_reps, warm_reps))

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["cache", "phase"],
            x_vals=[(cache, phase) for cache in ("cold", "warm") for phase in FIRST_LAUNCH_PHASES],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            ylabel=["time_ms"],
            plot_name="launch-overhead-cold",
            args={},
        ))
    def benchmark(cache, phase, provider):
        if provider != "triton":
            raise NotImplementedError(f"Unsupported provider {provider}")
        samples = measurements.get((), (cache, phase))[cache]
        _print_host_info_once(samples[0]["host"])
        times = [sample["phases_ms"][phase] for sample in samples]
        return (statistics.median(times), min(times), max(times)), _cv(times)

    return benchmark


def _worker_main():
    parser = argparse.ArgumentParser(description="Launch overhead worker: measures in this process, prints JSON.")
    parser.add_argument("--worker", choices=["launch", "first-launch"], required=True)
    parser.add_argument("--arg-type", choices=list(ARG_COUNTS), required=True)
    parser.add_argument("--num-args", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--n-launches", type=int)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--warmup-launches", type=int)
    args = parser.parse_args()
    if args.worker == "launch":
        result = measure_launch(args.arg_type, args.num_args, args.n_launches, args.reps, args.warmup_launches,
                                args.cpu)
    else:
        result = measure_first_launch(args.arg_type, args.num_args, args.cpu)
    print(json.dumps(result))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        get_benchmark().run(show_plots=False, print_data=True)
