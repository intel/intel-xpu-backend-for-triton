"""Performance benchmark for the extracted Triton layer-norm kernels 88/89/90.

The corresponding standalone dumps live under:
  - disassembles/inductor_test_log/j4/cj463y43ruy6tl3s5wlthtoqvvjtcelqwfskw6ib34d6vx7cqb2n.py
  - disassembles/inductor_test_log/d2/cd2gfxgpqns4l36pdxd4xsi3bwvelzncfo3uu2fgagnccsbobn6w.py
  - disassembles/inductor_test_log/xx/cxxp2y6ll76tdrgpzmxyfqvnvfuh56kokhsvyut6hhiuoaduduqb.py

Run examples:
  python triton_native_layer_norm_88_89_90_benchmark.py --kernel triton_red_fused_native_layer_norm_88 --xnumel 16384
  python triton_native_layer_norm_88_89_90_benchmark.py --kernel triton_poi_fused_native_layer_norm_90 --xnumel 1048576
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch

from triton_kernels_benchmark.benchmark_testing import DEVICE, get_do_bench

KERNELS_ROOT = Path(
    os.environ.get("TRITON_KERNEL_SCRIPT_DIR",
                   Path(__file__).resolve().parents[2] / "disassembles" / "inductor_test_log"))

KERNELS = {
    "triton_red_fused_native_layer_norm_88": {
        "path": KERNELS_ROOT / "j4" / "cj463y43ruy6tl3s5wlthtoqvvjtcelqwfskw6ib34d6vx7cqb2n.py",
        "kind": "reduction_mean",
    },
    "triton_red_fused_native_layer_norm_89": {
        "path": KERNELS_ROOT / "d2" / "cd2gfxgpqns4l36pdxd4xsi3bwvelzncfo3uu2fgagnccsbobn6w.py",
        "kind": "reduction_var",
    },
    "triton_poi_fused_native_layer_norm_90": {
        "path": KERNELS_ROOT / "xx" / "cxxp2y6ll76tdrgpzmxyfqvnvfuh56kokhsvyut6hhiuoaduduqb.py",
        "kind": "pointwise_layernorm",
    },
}

R0_NUMEL = 512


def _get_xpu_raw_stream():
    from torch._C import _xpu_getCurrentRawStream as get_raw_stream  # pylint: disable=C0415

    return get_raw_stream(torch.xpu.current_device())


def _resolve_kernel_path(kernel_name: str, override: str | None = None) -> Path:
    config = KERNELS[kernel_name]
    candidate = override or os.getenv(f"{kernel_name.upper().replace('-', '_').replace('.', '_')}_PATH")
    path = Path(candidate).expanduser() if candidate else config["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Could not find extracted kernel file '{path}' for {kernel_name}.")
    return path.resolve()


def _load_kernel(kernel_name: str, override: str | None = None):
    path = _resolve_kernel_path(kernel_name, override)
    spec = importlib.util.spec_from_file_location(f"{kernel_name}_bench_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import kernel '{kernel_name}' from '{path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, kernel_name):
        return getattr(module, kernel_name)
    raise AttributeError(f"Kernel '{kernel_name}' is not defined in '{path}'.")


def _make_reduction_inputs(xnumel: int, *, dtype: torch.dtype = torch.float16):
    torch.manual_seed(0)
    return torch.randn((xnumel, R0_NUMEL), device=DEVICE, dtype=dtype)


def _aten_layer_norm_reference(in0: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None):
    if weight is not None:
        weight = weight.to(torch.float32)
    if bias is not None:
        bias = bias.to(torch.float32)
    return torch.ops.aten.native_layer_norm(in0.to(torch.float32), [R0_NUMEL], weight, bias, 1e-6)


def _reference_reduction(kernel_name: str, in0: torch.Tensor) -> torch.Tensor:
    _, mean, rstd = _aten_layer_norm_reference(in0)
    if kernel_name == "triton_red_fused_native_layer_norm_88":
        return mean
    if kernel_name == "triton_red_fused_native_layer_norm_89":
        # aten native_layer_norm returns save_invstd = 1/sqrt(var + eps).
        return (1.0 / (rstd * rstd) - 1e-6)
    raise ValueError(f"Unsupported reduction kernel: {kernel_name}")


def _run_reduction_kernel(kernel, in0: torch.Tensor, out0: torch.Tensor, kernel_name: str) -> torch.Tensor:
    if DEVICE != "xpu":
        raise RuntimeError(f"{kernel_name} was extracted for XPU and cannot run on '{DEVICE}'.")
    kernel.run(in0, out0, in0.size(0), R0_NUMEL, stream=_get_xpu_raw_stream())
    return out0


def _make_pointwise_inputs(xnumel: int):
    if xnumel % R0_NUMEL != 0:
        raise ValueError(f"xnumel={xnumel} must be divisible by {R0_NUMEL} for this kernel.")
    rows = xnumel // R0_NUMEL
    torch.manual_seed(0)
    in0 = torch.randn((rows, R0_NUMEL), device=DEVICE, dtype=torch.float16)
    mean = in0.float().mean(dim=1)
    var = in0.float().var(dim=1, unbiased=False)
    weight = torch.randn((R0_NUMEL, ), device=DEVICE, dtype=torch.float16)
    bias = torch.randn((R0_NUMEL, ), device=DEVICE, dtype=torch.float16)
    return in0, mean, var, weight, bias


def _reference_pointwise(in0: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, weight: torch.Tensor,
                         bias: torch.Tensor):
    out, _, _ = _aten_layer_norm_reference(in0, weight, bias)
    return out.to(torch.float16).reshape(-1)


def _run_pointwise_kernel(kernel, in0: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, weight: torch.Tensor,
                          bias: torch.Tensor, out0: torch.Tensor, xnumel: int) -> torch.Tensor:
    if DEVICE != "xpu":
        raise RuntimeError(f"{kernel.__name__} was extracted for XPU and cannot run on '{DEVICE}'.")
    kernel.run(in0.reshape(-1), mean, var, weight, bias, out0, xnumel, stream=_get_xpu_raw_stream())
    return out0


def _flatten_output(tensor: torch.Tensor) -> torch.Tensor:
    """Keep the extracted-kernel flat output contract and ignore trailing singleton dims in the reference."""
    return tensor.reshape(-1)


def _reference_for_kernel(kernel_name: str, in0: torch.Tensor, mean: torch.Tensor | None = None,
                          var: torch.Tensor | None = None, weight: torch.Tensor | None = None,
                          bias: torch.Tensor | None = None):
    if kernel_name in {"triton_red_fused_native_layer_norm_88", "triton_red_fused_native_layer_norm_89"}:
        return _reference_reduction(kernel_name, in0)
    if kernel_name == "triton_poi_fused_native_layer_norm_90":
        assert mean is not None and var is not None and weight is not None and bias is not None
        return _reference_pointwise(in0, mean, var, weight, bias)
    raise ValueError(f"Unsupported kernel for reference path: {kernel_name}")


def _bench_kernel(kernel_name: str, xnumel: int, *, repeat: int = 5, warmup: int = 20, override: str | None = None):
    if kernel_name == "all":
        rows = []
        for name in sorted(KERNELS):
            rows.append(_bench_kernel(name, xnumel, repeat=repeat, warmup=warmup, override=override))
        return rows

    if kernel_name not in KERNELS:
        raise ValueError(f"Unsupported kernel {kernel_name}. Supported: {sorted(KERNELS)}")

    kernel = _load_kernel(kernel_name, override)
    do_bench = get_do_bench(n_warmup=warmup, n_repeat=repeat, quantiles=[0.5, 0.0, 1.0])

    if kernel_name in {"triton_red_fused_native_layer_norm_88", "triton_red_fused_native_layer_norm_89"}:
        in0 = _make_reduction_inputs(xnumel)
        out0 = torch.empty((xnumel, ), device=DEVICE, dtype=torch.float32)
        ref = _reference_reduction(kernel_name, in0)
        triton_fn = lambda: _run_reduction_kernel(kernel, in0, out0, kernel_name)
        aten_fn = lambda: _reference_for_kernel(kernel_name, in0)
        torch.testing.assert_close(_flatten_output(out0), _flatten_output(ref), rtol=1e-2, atol=1e-2)
    elif kernel_name == "triton_poi_fused_native_layer_norm_90":
        in0, mean, var, weight, bias = _make_pointwise_inputs(xnumel)
        out0 = torch.empty((xnumel, ), device=DEVICE, dtype=torch.float16)
        ref = _reference_pointwise(in0, mean, var, weight, bias)
        triton_fn = lambda: _run_pointwise_kernel(kernel, in0, mean, var, weight, bias, out0, xnumel)
        aten_fn = lambda: _reference_for_kernel(kernel_name, in0, mean, var, weight, bias)
        torch.testing.assert_close(_flatten_output(out0), _flatten_output(ref), rtol=1e-2, atol=1e-2)
    else:
        raise ValueError(f"Unsupported kernel for benchmark path: {kernel_name}")

    start = time.perf_counter()
    _ = do_bench(triton_fn)
    triton_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    _ = do_bench(aten_fn)
    aten_ms = (time.perf_counter() - start) * 1000.0

    return {
        "kernel": kernel_name,
        "xnumel": xnumel,
        "triton_ms": triton_ms,
        "aten_ms": aten_ms,
        "speedup": aten_ms / max(triton_ms, 1e-9),
        "elements": ref.numel(),
    }


def _run_aten_layer_norm(in0: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out, _, _ = _aten_layer_norm_reference(in0, weight, bias)
    return out.to(torch.float16).reshape(-1)


def _run_whole_layer_norm_pipeline(kernel88, kernel89, kernel90, in0: torch.Tensor, weight: torch.Tensor,
                                   bias: torch.Tensor, xnumel: int):
    rows = xnumel // R0_NUMEL
    mean = torch.empty((rows, ), device=DEVICE, dtype=torch.float32)
    var = torch.empty((rows, ), device=DEVICE, dtype=torch.float32)
    out0 = torch.empty((xnumel, ), device=DEVICE, dtype=torch.float16)
    _run_reduction_kernel(kernel88, in0, mean, "triton_red_fused_native_layer_norm_88")
    _run_reduction_kernel(kernel89, in0, var, "triton_red_fused_native_layer_norm_89")
    _run_pointwise_kernel(kernel90, in0, mean, var, weight, bias, out0, xnumel)
    return out0


def _bench_end_to_end(xnumel: int, *, repeat: int = 5, warmup: int = 20, override: str | None = None):
    kernel88 = _load_kernel("triton_red_fused_native_layer_norm_88", override)
    kernel89 = _load_kernel("triton_red_fused_native_layer_norm_89", override)
    kernel90 = _load_kernel("triton_poi_fused_native_layer_norm_90", override)
    do_bench = get_do_bench(n_warmup=warmup, n_repeat=repeat, quantiles=[0.5, 0.0, 1.0])

    in0, mean, var, weight, bias = _make_pointwise_inputs(xnumel)
    ref = _reference_pointwise(in0, mean, var, weight, bias)
    triton_fn = lambda: _run_whole_layer_norm_pipeline(kernel88, kernel89, kernel90, in0, weight, bias, xnumel)
    aten_fn = lambda: _run_aten_layer_norm(in0, weight, bias)

    out0 = _run_whole_layer_norm_pipeline(kernel88, kernel89, kernel90, in0, weight, bias, xnumel)
    torch.testing.assert_close(_flatten_output(out0), _flatten_output(ref), rtol=1e-2, atol=1e-2)

    start = time.perf_counter()
    _ = do_bench(triton_fn)
    triton_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    _ = do_bench(aten_fn)
    aten_ms = (time.perf_counter() - start) * 1000.0

    return {
        "kernel": "triton_88_89_90_pipeline",
        "xnumel": xnumel,
        "triton_ms": triton_ms,
        "aten_ms": aten_ms,
        "speedup": aten_ms / max(triton_ms, 1e-9),
        "elements": ref.numel(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the extracted Triton layer-norm kernels 88/89/90 as one full layer-norm pipeline.")
    parser.add_argument("--kernel", choices=sorted(KERNELS) + ["all", "pipeline"], default="pipeline")
    parser.add_argument("--xnumel", type=int, default=16384)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--path", type=str, default=None, help="Override the extracted kernel file path for --kernel.")
    args = parser.parse_args()

    if args.kernel == "pipeline":
        result = _bench_end_to_end(args.xnumel, repeat=args.repeat, warmup=args.warmup, override=args.path)
        print(f"kernel={result['kernel']} xnumel={result['xnumel']} "
              f"triton_ms={result['triton_ms']:.4f} aten_ms={result['aten_ms']:.4f} "
              f"speedup_vs_aten={result['speedup']:.3f}x")
        return

    if args.kernel == "all":
        results = _bench_kernel("all", args.xnumel, repeat=args.repeat, warmup=args.warmup, override=args.path)
        for result in results:
            print(f"kernel={result['kernel']} xnumel={result['xnumel']} "
                  f"triton_ms={result['triton_ms']:.4f} aten_ms={result['aten_ms']:.4f} "
                  f"speedup_vs_aten={result['speedup']:.3f}x")
        return

    result = _bench_kernel(args.kernel, args.xnumel, repeat=args.repeat, warmup=args.warmup, override=args.path)
    print(f"kernel={result['kernel']} xnumel={result['xnumel']} "
          f"triton_ms={result['triton_ms']:.4f} aten_ms={result['aten_ms']:.4f} "
          f"speedup_vs_aten={result['speedup']:.3f}x")
    if args.kernel in {"triton_red_fused_native_layer_norm_88", "triton_red_fused_native_layer_norm_89"}:
        bytes_per_elem = 2 + 4
    else:
        bytes_per_elem = 2 + 4 + 4 + 2 + 2 + 2
    print(f"estimated_gbps={(result['elements'] * bytes_per_elem * 1e-9) / (result['triton_ms'] / 1000.0):.3f}")


if __name__ == "__main__":
    main()
