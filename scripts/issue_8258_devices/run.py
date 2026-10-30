"""Compare stock Triton and SYCL-TLA for only the three original issue shapes."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


SHAPES = [[1, 8192, 6144, 12288], [1, 8192, 6144, 24576], [1, 2048, 6144, 24576]]
KERNEL_SHA256 = "b7381c83afe170d8a18ee3d40324fec6e2a3ce805e88694c0117e3871d9abe30"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(args, shape):
    import torch
    import triton
    import triton_kernels_benchmark as suite
    from triton_kernels_benchmark.benchmark_testing import MarkArgs

    if not suite.BENCHMARKING_CONFIG["verify"]:
        raise RuntimeError("Correctness verification must be enabled")
    device = torch.xpu.get_device_name()
    expected = {"b60": "B60", "max1550": "1550"}[args.device]
    if expected not in device:
        raise RuntimeError(f"Expected {args.device}, got {device}")
    source = args.checkout / "benchmarks/triton_kernels_benchmark/gemm_benchmark.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != KERNEL_SHA256:
        raise RuntimeError("Kernel source differs from the pinned B580 comparison")
    module_name = "triton_kernels_benchmark.gemm_benchmark"
    spec = importlib.util.spec_from_file_location(module_name, source)
    gemm = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = gemm
    spec.loader.exec_module(gemm)
    if gemm.sycl_tla_kernel is None or not hasattr(gemm.sycl_tla_kernel, "gemm"):
        raise RuntimeError("SYCL-TLA GEMM reference is required")
    if not gemm.is_enough_memory(shape):
        raise RuntimeError("Requested shape would be skipped for memory")
    gemm.X_VALS = [shape]
    benchmark = gemm.get_benchmark(providers_filter=["triton", "sycl-tla"])
    if benchmark.benchmarks.x_vals != [shape] or set(benchmark.benchmarks.line_vals) != {"triton", "sycl-tla"}:
        raise RuntimeError("Benchmark scope does not match the manifest")
    reference = Path(gemm.sycl_tla_kernel.__file__)
    write_json(args.reports / "provenance.json", {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "device": str(torch.xpu.get_device_properties()),
        "device_name": device,
        "requested_device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "benchmark_package": suite.__file__,
        "benchmark_wheel_run": 40643335150,
        "kernel_source_sha256": source_hash,
        "sycl_library": str(reference),
        "sycl_library_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "benchmarking_method": suite.BENCHMARKING_METHOD,
        "runner": os.getenv("RUNNER_NAME"),
        "shape": shape,
        "autotune_candidates": [str(c) for c in gemm.get_matmul_autotune_configs()],
        "repeats": 3,
        "verify": True,
        "environment": {k: os.getenv(k) for k in ("TRITON_XPU_GEN_NATIVE_CODE", "PTI_COLLECTION_MODE")},
    })
    try:
        frames = benchmark.run(
            show_plots=False, print_data=True, return_df=True,
            mark_args=MarkArgs(reports=str(args.reports), n_runs=3), save_precision=9,
        )
    finally:
        kernel = gemm.matmul_kernel_with_tensor_descriptors
        selected = [{"key": repr(k), "config": str(v)} for k, v in kernel.cache.items()]
        write_json(args.reports / "selected-configs.json", selected)
    if len(frames) != 3:
        raise RuntimeError("Missing benchmark repeats")
    times = {"triton_ms": [], "sycl_tla_ms": []}
    for frame in frames:
        measured = list(frame[["B", "M", "N", "K"]].itertuples(index=False, name=None))
        if measured != [tuple(shape)]:
            raise RuntimeError("Missing or extra GEMM shapes")
        for column, target in (("Triton-TFlops", "triton_ms"), ("SYCL-TLA-TFlops", "sycl_tla_ms")):
            throughput = float(frame.iloc[0][column])
            if not math.isfinite(throughput) or throughput <= 0:
                raise RuntimeError(f"Invalid throughput for {column}")
            times[target].append(2 * math.prod(shape) / (throughput * 1e9))
    medians = {key: statistics.median(values) for key, values in times.items()}
    write_json(args.reports / "result.json", {
        "shape": shape, "status": "passed", "times": times, "median": medians,
        "triton_extra_latency_pct": 100 * (medians["triton_ms"] / medians["sycl_tla_ms"] - 1),
    })


def run_cases(args, shapes):
    outcomes = []
    for index, shape in enumerate(shapes):
        report = args.reports / ("x".join(map(str, shape)))
        report.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--device", args.device,
                   "--checkout", str(args.checkout.resolve()), "--reports", str(report.resolve()),
                   "--worker", str(index)]
        print(f"Starting shape {shape}; output: {report / 'gemm.log'}", flush=True)
        with (report / "gemm.log").open("w") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2100)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = "timeout"
        result = report / "result.json"
        if returncode == 0 and result.exists():
            outcome = json.loads(result.read_text())
        else:
            outcome = {"shape": shape, "status": "failed", "returncode": returncode, "log": str(report / "gemm.log")}
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
        write_json(args.reports / "summary.json", {"device": args.device, "cases": outcomes})
    if any(case["status"] != "passed" for case in outcomes):
        raise RuntimeError("Some shapes failed; partial reports and logs were preserved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--device", choices=["b60", "max1550"], required=True)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--worker", type=int, choices=range(3), help=argparse.SUPPRESS)
    args = parser.parse_args()
    shapes = json.loads(Path(__file__).with_name("shapes.json").read_text())
    if shapes != SHAPES:
        raise ValueError("Manifest must contain exactly the three original issue shapes")
    if args.describe:
        print(json.dumps({"device": args.device, "shapes": shapes, "providers": ["triton", "sycl-tla"],
                          "repeats": 3, "verify": True, "policy": "stock autotuning and prefetch"}, indent=2))
        return
    args.reports.mkdir(parents=True, exist_ok=True)
    if args.worker is None:
        run_cases(args, shapes)
    else:
        worker(args, shapes[args.worker])


if __name__ == "__main__":
    main()
