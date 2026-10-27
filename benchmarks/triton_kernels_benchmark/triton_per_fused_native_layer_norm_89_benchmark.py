"""
Benchmark wrapper for the extracted Triton kernel
`triton_per_fused_native_layer_norm_89`.

The standalone dump for this implementation lives under
`disassembles/inductor_test_log/yc/cycrne7xmn3a7ejzggaqsq6j4c3uf45siigmriliav62pxdiwdhk.py`.
Some dumps refer to the same reduction kernel as
`triton_poi_fused_native_layer_norm_89`; other dumps use the adjacent
`triton_poi_fused_native_layer_norm_85` pointwise kernel, which is a different
operation and should not share this benchmark's reduction logic.
Set `TRITON_PER_FUSED_NATIVE_LAYER_NORM_89_PATH` to benchmark a different
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

KERNEL_NAME = "triton_per_fused_native_layer_norm_89"
KERNEL_ALIASES = (
    "triton_poi_fused_native_layer_norm_89",
    "triton_poi_fused_native_layer_norm_85",
)
KERNEL_PATH_ENV_VAR = "TRITON_PER_FUSED_NATIVE_LAYER_NORM_89_PATH"
R0_NUMEL = 512
DEFAULT_XNUMELS = [255, 256, 257, 1024, 4096, 16384]


def _get_default_kernel_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "disassembles" / "inductor_test_log" / "yc" / "cycrne7xmn3a7ejzggaqsq6j4c3uf45siigmriliav62pxdiwdhk.py"


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
    for name in (KERNEL_NAME, *KERNEL_ALIASES):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"'{kernel_path}' does not define '{KERNEL_NAME}' or one of {KERNEL_ALIASES}.")


def _get_xpu_raw_stream():
    from torch._C import _xpu_getCurrentRawStream as get_raw_stream  # pylint: disable=C0415

    return get_raw_stream(torch.xpu.current_device())


def _make_inputs(xnumel: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn((xnumel, R0_NUMEL), device=DEVICE, dtype=torch.float16)


def _make_outputs(xnumel: int) -> torch.Tensor:
    return torch.empty((xnumel, ), device=DEVICE, dtype=torch.float32)


def _run_reference(in0: torch.Tensor) -> torch.Tensor:
    centered = in0.float() - in0.float().mean(dim=1, keepdim=True)
    return centered.square().sum(dim=1)


def _run_triton(kernel, in0: torch.Tensor, out0: torch.Tensor) -> torch.Tensor:
    if DEVICE != "xpu":
        raise RuntimeError(f"{KERNEL_NAME} was extracted for XPU and cannot run on '{DEVICE}'.")
    kernel.run(in0, out0, in0.size(0), R0_NUMEL, stream=_get_xpu_raw_stream())
    return out0


def _perf_rate(work: float, ms: float) -> float:
    if ms == 0:
        return float("inf")
    return work / (ms * 1e-3)


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    kernel_path: Optional[str] = None,
    x_vals: Optional[list[int]] = None,
):
    """Build the benchmark report for the extracted layer-norm variance kernel."""

    supported_providers = {"torch": "Torch"}
    if DEVICE == "xpu":
        supported_providers = {"triton": "Triton", **supported_providers}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    resolved_kernel_path = _resolve_kernel_path(kernel_path) if "triton" in providers else None
    x_vals = x_vals if x_vals is not None else DEFAULT_XNUMELS

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["xnumel", "r0_numel"],
            x_vals=[(xnumel, R0_NUMEL) for xnumel in x_vals],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("blue", "-"), ("green", "--")],
            ylabel=["GB/s", "TFlops"],
            plot_name="triton-per-fused-native-layer-norm-89",
            args={},
        ))
    def benchmark(xnumel, r0_numel, provider):
        del r0_numel
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        in0 = _make_inputs(int(xnumel))

        if provider == "triton":
            if resolved_kernel_path is None:
                raise RuntimeError(f"{KERNEL_NAME} requires a standalone kernel path.")
            kernel = _load_kernel(str(resolved_kernel_path))
            triton_out = _make_outputs(int(xnumel))
            ref_out = _make_outputs(int(xnumel))
            triton_fn = lambda: _run_triton(kernel, in0, triton_out)
            torch_fn = lambda: _run_reference(in0)
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-3, rtol=1e-3, err_msg=KERNEL_NAME)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
        elif provider == "torch":
            torch_fn = lambda: _run_reference(in0)
            _, min_ms, max_ms, mean_ms, cv = do_bench(torch_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        total_bytes = int(xnumel) * (R0_NUMEL * 2 + 4)
        total_flops = int(xnumel) * (R0_NUMEL * 2)
        gbps = lambda ms: _perf_rate(total_bytes * 1e-9, ms)
        tflops = lambda ms: _perf_rate(total_flops * 1e-12, ms)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
