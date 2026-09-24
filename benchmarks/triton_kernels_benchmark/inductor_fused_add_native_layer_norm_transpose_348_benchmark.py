"""
Benchmark wrapper for the extracted TorchInductor Triton kernel
`triton_per_fused_add_native_layer_norm_transpose_348`.

By default this benchmark loads the standalone kernel dump at
`disassembles/inductor_test_log/d7/cd7hkxrlxelfsxmmltj7n3y2hdu7q3tk4iaracgslub74xwkjq22.py`.
Set the `TRITON_PER_FUSED_ADD_NATIVE_LAYER_NORM_TRANSPOSE_348_PATH`
environment variable to benchmark a different extracted kernel file.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path
from typing import Optional

import torch

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark.benchmark_testing import DEVICE

KERNEL_NAME = "triton_per_fused_add_native_layer_norm_transpose_348"
KERNEL_PATH_ENV_VAR = "TRITON_PER_FUSED_ADD_NATIVE_LAYER_NORM_TRANSPOSE_348_PATH"

X1 = 256
X0 = 80
R0 = 640
XNUMEL = X1 * X0


def _get_default_kernel_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "disassembles" / "inductor_test_log" / "d7" / "cd7hkxrlxelfsxmmltj7n3y2hdu7q3tk4iaracgslub74xwkjq22.py"


def _resolve_kernel_path(kernel_path: Optional[str]) -> Path:
    candidate = kernel_path or os.getenv(KERNEL_PATH_ENV_VAR)
    path = Path(candidate).expanduser() if candidate else _get_default_kernel_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find extracted kernel file '{path}'. Set {KERNEL_PATH_ENV_VAR} to the standalone kernel path.")
    return path.resolve()


@functools.lru_cache(maxsize=None)
def _load_kernel(kernel_path: str):
    spec = importlib.util.spec_from_file_location(f"{KERNEL_NAME}_module", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create an import spec for '{kernel_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, KERNEL_NAME)
    except AttributeError as exc:
        raise AttributeError(f"'{kernel_path}' does not define '{KERNEL_NAME}'.") from exc


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    in0 = torch.randn((X1, X0, R0), device=DEVICE, dtype=torch.float16)
    in1 = torch.randn((X0, X1, R0), device=DEVICE, dtype=torch.float16)
    in2 = torch.randn((X0, 1, R0), device=DEVICE, dtype=torch.float16)
    return in0, in1, in2


def _make_outputs() -> tuple[torch.Tensor, torch.Tensor]:
    out0 = torch.empty_strided((X1, X0, 1), (X0, 1, XNUMEL), device=DEVICE, dtype=torch.float32)
    out1 = torch.empty_strided((X1, X0, 1), (X0, 1, XNUMEL), device=DEVICE, dtype=torch.float32)
    return out0, out1


def _run_reference(in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, out0: torch.Tensor,
                   out1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = in0.float() + in1.permute(1, 0, 2).float() + in2.permute(1, 0, 2).float()
    mean = values.mean(dim=2, keepdim=True)
    centered = values - mean
    out0.copy_(mean)
    out1.copy_((centered * centered).sum(dim=2, keepdim=True))
    return out0, out1


def _get_xpu_raw_stream():
    from torch._C import _xpu_getCurrentRawStream as get_raw_stream  # pylint: disable=C0415

    return get_raw_stream(torch.xpu.current_device())


def _run_triton(kernel, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, out0: torch.Tensor,
                out1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if DEVICE != "xpu":
        raise RuntimeError(f"{KERNEL_NAME} was extracted for XPU and cannot run on '{DEVICE}'.")
    kernel.run(in0, in1, in2, out0, out1, XNUMEL, R0, stream=_get_xpu_raw_stream())
    return out0, out1


def _perf_rate(work: float, ms: float) -> float:
    if ms == 0:
        return float("inf")
    return work / (ms * 1e-3)


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    kernel_path: Optional[str] = None,
):
    """Build the benchmark report for the extracted fused add + layer-norm stats kernel."""

    supported_providers = {"torch": "Torch"}
    if DEVICE == "xpu":
        supported_providers = {"triton": "Triton", **supported_providers}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    resolved_kernel_path = _resolve_kernel_path(kernel_path) if "triton" in providers else None

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["x1", "x0", "r0"],
            x_vals=[(X1, X0, R0)],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("blue", "-"), ("green", "--")],
            ylabel=["GB/s", "TFlops"],
            plot_name="inductor-fused-add-native-layer-norm-transpose-348",
            args={},
        ))
    def benchmark(x1, x0, r0, provider):
        del x1, x0, r0
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        in0, in1, in2 = _make_inputs()

        if provider == "triton":
            if resolved_kernel_path is None:
                raise RuntimeError(f"{KERNEL_NAME} requires a standalone kernel path.")
            kernel = _load_kernel(str(resolved_kernel_path))
            triton_out0, triton_out1 = _make_outputs()
            ref_out0, ref_out1 = _make_outputs()
            triton_fn = lambda: _run_triton(kernel, in0, in1, in2, triton_out0, triton_out1)
            torch_fn = lambda: _run_reference(in0, in1, in2, ref_out0, ref_out1)
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-3, rtol=1e-3, err_msg=KERNEL_NAME)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
        elif provider == "torch":
            ref_out0, ref_out1 = _make_outputs()
            torch_fn = lambda: _run_reference(in0, in1, in2, ref_out0, ref_out1)
            _, min_ms, max_ms, mean_ms, cv = do_bench(torch_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        total_bytes = XNUMEL * (R0 * (3 * 2) + 2 * 4)
        total_flops = XNUMEL * (6 * R0 - 1)
        gbps = lambda ms: _perf_rate(total_bytes * 1e-9, ms)
        tflops = lambda ms: _perf_rate(total_flops * 1e-12, ms)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
