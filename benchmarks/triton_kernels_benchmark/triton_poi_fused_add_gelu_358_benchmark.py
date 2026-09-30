"""
Benchmark wrapper for the standalone Triton kernel
`triton_poi_fused_add_gelu_358` extracted from Inductor output.

By default this benchmark loads the standalone kernel dump at
`disassembles/inductor_test_log/yp/cyptkx4fx7yvuj7o74ec6fopi4khkagiqdygza6cxqlr2yfpoql3.py`.
Set `TRITON_POI_FUSED_ADD_GELU_358_PATH` to point at another extracted
kernel file when needed.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark.benchmark_testing import DEVICE

KERNEL_NAME = "triton_poi_fused_add_gelu_358"
KERNEL_PATH_ENV_VAR = "TRITON_POI_FUSED_ADD_GELU_358_PATH"
DEFAULT_SHAPES = [(16, 256, 160)]
SECONDARY_SHAPE = (16, 1, 160)


def _get_default_kernel_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "disassembles" / "inductor_test_log" / "yp" / "cyptkx4fx7yvuj7o74ec6fopi4khkagiqdygza6cxqlr2yfpoql3.py"


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


def _get_xpu_raw_stream():
    from torch._C import _xpu_getCurrentRawStream as get_raw_stream  # pylint: disable=C0415

    return get_raw_stream(torch.xpu.current_device())


def _validate_numel(r0_numel: int) -> int:
    return r0_numel


def _make_inputs(shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, cols = shape
    if rows <= 0 or cols <= 0 or batch <= 0:
        raise ValueError(f"Invalid input shape {shape} for {KERNEL_NAME}.")
    if cols != SECONDARY_SHAPE[2] or batch != SECONDARY_SHAPE[0]:
        raise ValueError(f"Expected shape {shape} to match the kernel's broadcast pattern ({batch}, {rows}, {cols})")
    torch.manual_seed(0)
    in_out = torch.randn((batch, rows, cols), device=DEVICE, dtype=torch.float16)
    in_ptr0 = torch.randn((batch, 1, cols), device=DEVICE, dtype=torch.float16)
    return in_out, in_ptr0


def _run_reference(in_out: torch.Tensor, in_ptr0: torch.Tensor) -> torch.Tensor:
    return F.gelu(in_out.to(torch.float32) + in_ptr0.to(torch.float32), approximate="none").to(torch.float16)


def _run_triton(
    kernel,
    in_out: torch.Tensor,
    in_out_template: torch.Tensor,
    in_ptr0: torch.Tensor,
    xnumel: int,
) -> torch.Tensor:
    if DEVICE != "xpu":
        raise RuntimeError(f"{KERNEL_NAME} was extracted for XPU and cannot run on '{DEVICE}'.")
    in_out.copy_(in_out_template)
    in_out_flat = in_out.reshape(-1)
    in_ptr0_flat = in_ptr0.reshape(-1)
    kernel.run(in_out_flat, in_ptr0_flat, xnumel, stream=_get_xpu_raw_stream())
    return in_out


def _perf_rate(work: float, ms: float) -> float:
    if ms == 0:
        return float("inf")
    return work / (ms * 1e-3)


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    kernel_path: Optional[str] = None,
    x_vals: Optional[list[int]] = None,
):
    """Build a benchmark report for the extracted add+GELU Triton kernel."""

    supported_providers = {"torch": "Torch"}
    if DEVICE == "xpu":
        supported_providers = {"triton": "Triton", **supported_providers}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    resolved_kernel_path = _resolve_kernel_path(kernel_path) if "triton" in providers else None
    x_vals = x_vals if x_vals is not None else DEFAULT_SHAPES

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["batch", "rows", "cols"],
            x_vals=[shape for shape in x_vals],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("green", "-"), ("blue", "--")],
            ylabel=["GB/s", "TFlops"],
            plot_name="triton-poi-fused-add-gelu-358",
            args={},
        ))
    def benchmark(batch, rows, cols, provider):
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        in_out_template, in_ptr0 = _make_inputs((batch, rows, cols))

        if provider == "triton":
            if resolved_kernel_path is None:
                raise RuntimeError(f"{KERNEL_NAME} requires a standalone kernel path.")
            kernel = _load_kernel(str(resolved_kernel_path))
            triton_out = in_out_template.clone()
            ref_out = _run_reference(in_out_template, in_ptr0)

            triton_fn = lambda: _run_triton(kernel, triton_out, in_out_template, in_ptr0, batch * rows * cols)
            torch_fn = lambda: ref_out
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-3, rtol=1e-3, err_msg=KERNEL_NAME)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
        elif provider == "torch":
            torch_fn = lambda: _run_reference(in_out_template, in_ptr0)
            _, min_ms, max_ms, mean_ms, cv = do_bench(torch_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        total_bytes = batch * rows * cols * (2 + 2 + 2)
        total_flops = batch * rows * cols * 6
        gbps = lambda ms: _perf_rate(total_bytes * 1e-9, ms)
        tflops = lambda ms: _perf_rate(total_flops * 1e-12, ms)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
