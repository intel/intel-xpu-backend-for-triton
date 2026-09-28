"""
Benchmark wrapper for the standalone Triton kernel
`triton_red_fused__to_copy_cumsum_96` extracted from Inductor output.

By default this benchmark loads the standalone kernel dump at
`disassembles/inductor_test_log/6r/c6rh2xkecdcc4zbij57iakmxka2z4xaarvk4c3ko2wtgwjsepdex.py`.
Set `TRITON_RED_FUSED_TO_COPY_CUMSUM_96_PATH` to point at another extracted
kernel file when needed.
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

KERNEL_NAME = "triton_red_fused__to_copy_cumsum_96"
KERNEL_PATH_ENV_VAR = "TRITON_RED_FUSED_TO_COPY_CUMSUM_96_PATH"
DEFAULT_SHAPES = [256, 1024, 4096, 16384, 65536, 289239, 33699, 11793]


def _get_default_kernel_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "disassembles" / "inductor_test_log" / "6r" / "c6rh2xkecdcc4zbij57iakmxka2z4xaarvk4c3ko2wtgwjsepdex.py"


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


def _make_inputs(r0_numel: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 2, (r0_numel, ), device=DEVICE, dtype=torch.bool)


def _run_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x.to(torch.int64), dim=0)


def _run_triton(kernel, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if DEVICE != "xpu":
        raise RuntimeError(f"{KERNEL_NAME} was extracted for XPU and cannot run on '{DEVICE}'.")
    kernel.run(x, out, 1, x.numel(), stream=_get_xpu_raw_stream())
    return out


def _perf_rate(work: float, ms: float) -> float:
    if ms == 0:
        return float("inf")
    return work / (ms * 1e-3)


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    kernel_path: Optional[str] = None,
    x_vals: Optional[list[int]] = None,
):
    """Build a benchmark report for the extracted cumsum Triton kernel."""

    supported_providers = {"torch": "Torch"}
    if DEVICE == "xpu":
        supported_providers = {"triton": "Triton", **supported_providers}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    resolved_kernel_path = _resolve_kernel_path(kernel_path) if "triton" in providers else None
    x_vals = x_vals if x_vals is not None else DEFAULT_SHAPES

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["r0"],
            x_vals=[(r0, ) for r0 in x_vals],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("green", "-"), ("blue", "--")],
            ylabel=["GB/s", "TFlops"],
            plot_name="triton-red-fused-to-copy-cumsum-96",
            args={},
        ))
    def benchmark(r0, provider):
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        x = _make_inputs(int(r0))

        if provider == "triton":
            if resolved_kernel_path is None:
                raise RuntimeError(f"{KERNEL_NAME} requires a standalone kernel path.")
            kernel = _load_kernel(str(resolved_kernel_path))
            triton_out = torch.empty((r0, ), device=DEVICE, dtype=torch.int64)
            ref_out = torch.empty((r0, ), device=DEVICE, dtype=torch.int64)

            triton_fn = lambda: _run_triton(kernel, x, triton_out)
            torch_fn = lambda: _run_reference(x).to(device=DEVICE, dtype=torch.int64)
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=0, rtol=0, err_msg=KERNEL_NAME)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
        elif provider == "torch":
            torch_fn = lambda: _run_reference(x).to(device=DEVICE, dtype=torch.int64)
            _, min_ms, max_ms, mean_ms, cv = do_bench(torch_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        total_bytes = r0 * (1 + 8)
        total_flops = r0
        gbps = lambda ms: _perf_rate(total_bytes * 1e-9, ms)
        tflops = lambda ms: _perf_rate(total_flops * 1e-12, ms)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
