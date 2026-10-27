"""Standalone benchmarks for the extracted layer-norm mean (88) and M2 (89) reductions.

Run from the repository root:
  PYTHONPATH=benchmarks python3 -m triton_kernels_benchmark.triton_red_fused_native_layer_norm_88_benchmark

Each input has shape [xnumel, 512], where xnumel is the number of rows.
Kernel 88 returns the row mean; kernel 89 returns sum((x - mean)**2),
not the population variance. Both outputs are FP32.

Set TRITON_KERNEL_SCRIPT_DIR to override the dump root, or set
TRITON_RED_FUSED_NATIVE_LAYER_NORM_88_PATH / TRITON_RED_FUSED_NATIVE_LAYER_NORM_89_PATH
to override an individual dump. get_benchmark also accepts kernel_path and x_vals.
Reports show logical input/output bandwidth and measured latency, excluding
input/output allocation and kernel import/autotuning.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path
from typing import Any

import torch

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark.benchmark_testing import DEVICE

KERNEL_NAME = "triton_red_fused_native_layer_norm_88"
R0_NUMEL = 512
DEFAULT_XNUMELS = [255, 256, 257, 1024, 4096, 16384]
KERNEL_PATHS = {
    KERNEL_NAME: Path("j4/cj463y43ruy6tl3s5wlthtoqvvjtcelqwfskw6ib34d6vx7cqb2n.py"),
    "triton_red_fused_native_layer_norm_89": Path("d2/cd2gfxgpqns4l36pdxd4xsi3bwvelzncfo3uu2fgagnccsbobn6w.py"),
}


def _resolve_kernel_path(kernel_name: str, kernel_path: str | None) -> Path:
    env_var = f"{kernel_name.upper()}_PATH"
    candidate = kernel_path or os.getenv(env_var)
    root = Path(
        os.environ.get("TRITON_KERNEL_SCRIPT_DIR",
                       Path(__file__).resolve().parents[2] / "disassembles" / "inductor_test_log")).expanduser()
    path = Path(candidate).expanduser() if candidate else root / KERNEL_PATHS[kernel_name]
    if not path.is_file():
        raise FileNotFoundError(f"Could not find extracted kernel '{path}'. Set {env_var} or pass kernel_path.")
    return path.resolve()


@functools.lru_cache(maxsize=None)
def _load_kernel(kernel_name: str, kernel_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{kernel_name}_bench_module", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create an import spec for '{kernel_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, kernel_name):
        raise AttributeError(f"'{kernel_path}' does not define '{kernel_name}'.")
    return getattr(module, kernel_name)


def _run_reference(in0: torch.Tensor, out0: torch.Tensor, *, compute_m2: bool) -> torch.Tensor:
    x = in0.float()
    if compute_m2:
        out0.copy_(x.var(dim=1, unbiased=False) * R0_NUMEL)
    else:
        out0.copy_(x.mean(dim=1))
    return out0


def _run_triton(kernel: Any, in0: torch.Tensor, out0: torch.Tensor) -> torch.Tensor:
    if DEVICE != "xpu":
        raise RuntimeError(f"The extracted reduction kernels require XPU, not '{DEVICE}'.")
    from torch._C import _xpu_getCurrentRawStream  # pylint: disable=C0415

    stream = _xpu_getCurrentRawStream(torch.xpu.current_device())
    kernel.run(in0, out0, in0.size(0), R0_NUMEL, stream=stream)
    return out0


def _get_benchmark(
    kernel_name: str,
    providers_filter: list[str] | None = None,
    kernel_path: str | None = None,
    x_vals: list[int] | None = None,
) -> Any:
    supported_providers = {"torch": "Torch"}
    if DEVICE == "xpu":
        supported_providers = {"triton": "Triton", **supported_providers}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    resolved_path = _resolve_kernel_path(kernel_name, kernel_path) if "triton" in providers else None
    x_vals = list(DEFAULT_XNUMELS if x_vals is None else x_vals)
    if not x_vals or any(not isinstance(rows, int) or rows <= 0 for rows in x_vals):
        raise ValueError("x_vals must contain positive integer row counts.")
    compute_m2 = kernel_name == "triton_red_fused_native_layer_norm_89"

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["xnumel", "r0_numel"],
            x_vals=[(rows, R0_NUMEL) for rows in x_vals],
            line_arg="provider",
            line_vals=list(providers),
            line_names=list(providers.values()),
            styles=[("blue", "-"), ("green", "--")],
            ylabel=["GB/s", "ms"],
            plot_name=kernel_name.replace("_", "-"),
            args={},
        ))
    def benchmark(xnumel: int, r0_numel: int, provider: str) -> tuple:
        torch.manual_seed(0)
        in0 = torch.randn((xnumel, r0_numel), device=DEVICE, dtype=torch.float16)
        ref_out = torch.empty((xnumel, ), device=DEVICE, dtype=torch.float32)
        torch_fn = functools.partial(_run_reference, in0, ref_out, compute_m2=compute_m2)
        if provider == "triton":
            if resolved_path is None:
                raise RuntimeError(f"{kernel_name} requires a standalone kernel path.")
            kernel = _load_kernel(kernel_name, str(resolved_path))
            triton_out = torch.empty_like(ref_out)
            run = functools.partial(_run_triton, kernel, in0, triton_out)
            benchmark_suite.assert_close(run, torch_fn, atol=1e-3, rtol=1e-3, err_msg=kernel_name)
        elif provider == "torch":
            run = torch_fn
        else:
            raise NotImplementedError(f"Unsupported provider '{provider}'.")

        # Initialize/autotune before entering the timing harness, even with VERIFY=0.
        run()
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        _, min_ms, max_ms, mean_ms, cv = do_bench(run)
        total_bytes = in0.numel() * in0.element_size() + ref_out.numel() * ref_out.element_size()

        def gbps(ms: float) -> float:
            return total_bytes * 1e-6 / ms if ms > 0 else float("inf")

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (mean_ms, min_ms, max_ms), cv

    return benchmark


def get_benchmark(
    providers_filter: list[str] | None = None,
    kernel_path: str | None = None,
    x_vals: list[int] | None = None,
) -> Any:
    """Build the mean-reduction report for FP16 rows of length 512.

    Arguments:
      providers_filter: Providers to benchmark ("triton" and/or "torch").
      kernel_path: Optional extracted kernel dump override.
      x_vals: Positive row counts; defaults include partial-block cases.

    Returns:
      A benchmark-suite report with bandwidth and latency metrics.
    """
    return _get_benchmark(KERNEL_NAME, providers_filter, kernel_path, x_vals)


if __name__ == "__main__":
    get_benchmark().run(show_plots=False, print_data=True)
