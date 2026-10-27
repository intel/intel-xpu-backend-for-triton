"""Standalone benchmark for extracted Triton layer-norm kernels 91/92/93."""
from __future__ import annotations
import argparse
import importlib.util
import os
import time
from pathlib import Path
import torch
from triton_kernels_benchmark.benchmark_testing import DEVICE, get_do_bench

KERNELS = {
    "triton_per_fused_native_layer_norm_91": {
        "path":
        Path(__file__).resolve().parents[2] / "disassembles" / "inductor_test_log" / "wj" /
        "cwjx57tzm7dbfqocsahk2rmwnf4j7zbsugw2mwoqaxdenascpteb.py",
    },
    "triton_per_fused_native_layer_norm_92": {
        "path":
        Path(__file__).resolve().parents[2] / "disassembles" / "inductor_test_log" / "bf" /
        "cbf6dz7bdcz3ldu4gmygcbub47qf6awqxseuedqqa76tsqzdkzzm.py",
    },
    "triton_poi_fused_native_layer_norm_93": {
        "path":
        Path(__file__).resolve().parents[2] / "disassembles" / "inductor_test_log" / "ka" /
        "cka2uvewroxreusiycrgatotjea6shfp6pofe6h4gasokyj3xwc5.py",
    },
}
R0_NUMEL = 512


def _get_xpu_raw_stream():
    from torch._C import _xpu_getCurrentRawStream as get_raw_stream  # pylint: disable=C0415
    return get_raw_stream(torch.xpu.current_device())


def _load_kernel(kernel_name: str, override: str | None = None):
    candidate = override or os.getenv(f"{kernel_name.upper().replace('-', '_').replace('.', '_')}_PATH")
    path = Path(candidate).expanduser() if candidate else KERNELS[kernel_name]["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Could not find kernel dump: {path}")
    spec = importlib.util.spec_from_file_location(kernel_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {kernel_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, kernel_name):
        return getattr(mod, kernel_name)
    raise AttributeError(f"{kernel_name} not found in {path}")


def _make_reduction_inputs(xnumel: int):
    torch.manual_seed(0)
    return torch.randn((xnumel, R0_NUMEL), device=DEVICE, dtype=torch.float16)


def _make_pointwise_inputs(xnumel: int):
    if xnumel % R0_NUMEL != 0:
        raise ValueError(f"xnumel must be divisible by {R0_NUMEL} for 91/92/93 benchmarks")
    rows = xnumel // R0_NUMEL
    torch.manual_seed(0)
    in0 = torch.randn((rows, R0_NUMEL), device=DEVICE, dtype=torch.float16)
    mean = in0.float().mean(dim=1)
    var = in0.float().var(dim=1, unbiased=False)
    weight = torch.randn((R0_NUMEL, ), device=DEVICE, dtype=torch.float16)
    bias = torch.randn((R0_NUMEL, ), device=DEVICE, dtype=torch.float16)
    return in0, mean, var, weight, bias


def _run_reduction(kernel, in0: torch.Tensor, out0: torch.Tensor):
    if DEVICE != "xpu":
        raise RuntimeError(f"Kernel extracted for XPU only; active device is {DEVICE}")
    kernel.run(in0, out0, in0.size(0), R0_NUMEL, stream=_get_xpu_raw_stream())


def _run_pointwise(kernel, in0: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, weight: torch.Tensor,
                   bias: torch.Tensor, out0: torch.Tensor, xnumel: int):
    if DEVICE != "xpu":
        raise RuntimeError(f"Kernel extracted for XPU only; active device is {DEVICE}")
    kernel.run(in0.reshape(-1), mean, var, weight, bias, out0, xnumel, stream=_get_xpu_raw_stream())


def _reference_mean(in0: torch.Tensor):
    return in0.float().mean(dim=1)


def _reference_var(in0: torch.Tensor):
    return in0.float().var(dim=1, unbiased=False)


def _reference_pointwise(in0: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, weight: torch.Tensor,
                         bias: torch.Tensor):
    x = in0.to(torch.float32)
    y = ((x - mean[:, None]) * torch.rsqrt(var[:, None] / float(R0_NUMEL) + 1e-6))
    y = y * weight[None, :].to(torch.float32)
    y = y + bias[None, :].to(torch.float32)
    return y.to(torch.float16).reshape(-1)


def benchmark_one(kernel_name: str, xnumel: int, repeat: int = 10, warmup: int = 20, override: str | None = None):
    kernel = _load_kernel(kernel_name, override)
    bench = get_do_bench(n_warmup=warmup, n_repeat=repeat)
    if kernel_name in {"triton_per_fused_native_layer_norm_91", "triton_per_fused_native_layer_norm_92"}:
        in0 = _make_reduction_inputs(xnumel)
        out0 = torch.empty((xnumel, ), device=DEVICE, dtype=torch.float32)
        ref = _reference_mean(in0) if kernel_name.endswith("91") else _reference_var(in0)
        fn = lambda: _run_reduction(kernel, in0, out0)
        torch.testing.assert_close(out0, ref, rtol=1e-2, atol=1e-2)
    elif kernel_name == "triton_poi_fused_native_layer_norm_93":
        in0, mean, var, weight, bias = _make_pointwise_inputs(xnumel)
        out0 = torch.empty((xnumel, ), device=DEVICE, dtype=torch.float16)
        ref = _reference_pointwise(in0, mean, var, weight, bias)
        fn = lambda: _run_pointwise(kernel, in0, mean, var, weight, bias, out0, xnumel)
        torch.testing.assert_close(out0, ref, rtol=1e-2, atol=1e-2)
    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")
    t0 = time.perf_counter()
    _ = bench(fn)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"kernel={kernel_name} xnumel={xnumel} total_ms={elapsed_ms:.3f} avg_ms={elapsed_ms / max(repeat, 1):.3f}")
    return elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark 91/92/93 extracted layer-norm kernels")
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="triton_per_fused_native_layer_norm_91")
    parser.add_argument("--xnumel", type=int, default=16384)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    benchmark_one(args.kernel, args.xnumel, repeat=args.repeat, warmup=args.warmup, override=args.path)


if __name__ == "__main__":
    main()
