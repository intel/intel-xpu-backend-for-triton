"""
Benchmark wrapper for the extracted Triton kernel
`triton_per_fused__to_copy_amax_gt_view_105`.

By default this benchmark loads the standalone kernel dump at
`disassembles/inductor_test_log/7x/c7x6ntvrpdoyah5orgyoy2jk7kpnvxxknbe3o5ef3m55hhrzh2uc.py`.
Set `TRITON_PER_FUSED_TO_COPY_AMAX_GT_VIEW_105_PATH` to benchmark a different
extracted kernel file.
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

KERNEL_NAME = "triton_per_fused__to_copy_amax_gt_view_105"
KERNEL_PATH_ENV_VAR = "TRITON_PER_FUSED_TO_COPY_AMAX_GT_VIEW_105_PATH"
XNUMEL = 22016
R0_NUMEL = 8


def _get_default_kernel_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "disassembles" / "inductor_test_log" / "7x" / "c7x6ntvrpdoyah5orgyoy2jk7kpnvxxknbe3o5ef3m55hhrzh2uc.py"


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


def _make_inputs() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(-4, 5, (XNUMEL, R0_NUMEL), device=DEVICE, dtype=torch.int64)


def _make_outputs() -> tuple[torch.Tensor, torch.Tensor]:
    out0 = torch.empty((XNUMEL, ), device=DEVICE, dtype=torch.int64)
    out1 = torch.empty((XNUMEL, ), device=DEVICE, dtype=torch.int32)
    return out0, out1


def _run_reference(in0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The extracted kernel computes the max over the per-row `> 0` predicate, then
    # stores a boolean mask based on whether any value in the row is positive.
    pred = in0 > 0
    out0 = pred.to(torch.int64).amax(dim=1)
    out1 = (out0 > 0).to(torch.int32)
    return out0, out1


def _run_triton(kernel, in0: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if DEVICE != "xpu":
        raise RuntimeError(f"{KERNEL_NAME} was extracted for XPU and cannot run on '{DEVICE}'.")
    kernel.run(in0, out0, out1, XNUMEL, R0_NUMEL, stream=_get_xpu_raw_stream())
    return out0, out1


def _perf_rate(work: float, ms: float) -> float:
    if ms == 0:
        return float("inf")
    return work / (ms * 1e-3)


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    kernel_path: Optional[str] = None,
):
    """Build the benchmark report for the extracted amax-or-bool Triton kernel."""

    supported_providers = {"torch": "Torch"}
    if DEVICE == "xpu":
        supported_providers = {"triton": "Triton", **supported_providers}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    resolved_kernel_path = _resolve_kernel_path(kernel_path) if "triton" in providers else None

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["xnumel", "r0_numel"],
            x_vals=[(XNUMEL, R0_NUMEL)],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("blue", "-"), ("green", "--")],
            ylabel=["GB/s", "TFlops"],
            plot_name="triton-per-fused-to-copy-amax-gt-view-105",
            args={},
        ))
    def benchmark(xnumel, r0_numel, provider):
        del xnumel, r0_numel
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        in0 = _make_inputs()

        if provider == "triton":
            if resolved_kernel_path is None:
                raise RuntimeError(f"{KERNEL_NAME} requires a standalone kernel path.")
            kernel = _load_kernel(str(resolved_kernel_path))
            triton_out0, triton_out1 = _make_outputs()
            ref_out0, ref_out1 = _run_reference(in0)
            triton_fn = lambda: _run_triton(kernel, in0, triton_out0, triton_out1)
            torch_fn = lambda: _run_reference(in0)
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=0, rtol=0, err_msg=KERNEL_NAME)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
            del ref_out0, ref_out1
        elif provider == "torch":
            torch_fn = lambda: _run_reference(in0)
            _, min_ms, max_ms, mean_ms, cv = do_bench(torch_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        total_bytes = XNUMEL * (R0_NUMEL * 8 + 8 + 4)
        total_flops = XNUMEL * (R0_NUMEL + (R0_NUMEL - 1))
        gbps = lambda ms: _perf_rate(total_bytes * 1e-9, ms)
        tflops = lambda ms: _perf_rate(total_flops * 1e-12, ms)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
