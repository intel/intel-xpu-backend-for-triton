"""Run only the issue-8258 GEMM manifest; --describe needs no GPU libraries."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    args = parser.parse_args()
    shapes = json.loads(Path(__file__).with_name("shapes.json").read_text())
    if len(shapes) != 12 or len({tuple(s) for s in shapes}) != 12:
        raise ValueError("Expected exactly twelve unique GEMM shapes")
    if any(len(s) != 4 or s[0] != 1 or any(type(v) is not int or v <= 0 for v in s) for s in shapes):
        raise ValueError("Expected positive B=1,M,N,K shapes")
    if args.describe:
        print(json.dumps({"shapes": shapes, "providers": ["triton", "sycl-tla"], "repeats": 3}, indent=2))
        return

    import torch
    import triton
    import triton_kernels_benchmark as suite
    from triton_kernels_benchmark.benchmark_testing import MarkArgs

    if not suite.BENCHMARKING_CONFIG["verify"]:
        raise RuntimeError("Correctness verification must be enabled")
    if "B580" not in torch.xpu.get_device_name():
        raise RuntimeError("This investigation requires a B580 runner")
    source = args.checkout / "benchmarks/triton_kernels_benchmark/gemm_benchmark.py"
    module_name = "triton_kernels_benchmark.gemm_benchmark"
    spec = importlib.util.spec_from_file_location(module_name, source)
    gemm = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = gemm
    spec.loader.exec_module(gemm)
    if gemm.sycl_tla_kernel is None or not hasattr(gemm.sycl_tla_kernel, "gemm"):
        raise RuntimeError("SYCL-TLA GEMM reference is required")
    if not all(gemm.is_enough_memory(s) for s in shapes):
        raise RuntimeError("Requested shape would be skipped for memory")
    gemm.X_VALS = shapes
    benchmark = gemm.get_benchmark(providers_filter=["triton", "sycl-tla"])
    if benchmark.benchmarks.x_vals != shapes or set(benchmark.benchmarks.line_vals) != {"triton", "sycl-tla"}:
        raise RuntimeError("Benchmark scope does not match the manifest")

    args.reports.mkdir(parents=True, exist_ok=True)
    reference = Path(gemm.sycl_tla_kernel.__file__)
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "device": str(torch.xpu.get_device_properties()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "benchmark_package": suite.__file__,
        "kernel_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sycl_library": str(reference),
        "sycl_library_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "benchmarking_method": suite.BENCHMARKING_METHOD,
        "runner": os.getenv("RUNNER_NAME"),
        "shapes": shapes,
        "autotune_candidates": [str(c) for c in gemm.get_matmul_autotune_configs()],
        "repeats": 3,
        "verify": True,
    }
    (args.reports / "provenance.json").write_text(json.dumps(provenance, indent=2))
    frames = benchmark.run(
        show_plots=False, print_data=True, return_df=True,
        mark_args=MarkArgs(reports=str(args.reports), n_runs=3), save_precision=9,
    )
    if len(frames) != 3:
        raise RuntimeError("Missing benchmark repeats")
    requested = {tuple(s) for s in shapes}
    for frame in frames:
        measured = {tuple(s) for s in frame[["B", "M", "N", "K"]].itertuples(index=False, name=None)}
        if len(frame) != 12 or measured != requested:
            raise RuntimeError("Missing or extra GEMM shapes")
        for name in ("Triton-TFlops", "SYCL-TLA-TFlops"):
            if not all(math.isfinite(float(v)) and float(v) > 0 for v in frame[name]):
                raise RuntimeError(f"Invalid throughput for {name}")
    kernel = gemm.matmul_kernel_with_tensor_descriptors
    selected = [{"key": repr(k), "config": str(v)} for k, v in kernel.cache.items()]
    (args.reports / "selected-configs.json").write_text(json.dumps(selected, indent=2))
    print("Verified twelve shapes, two providers, and three complete repeats.")


if __name__ == "__main__":
    main()
