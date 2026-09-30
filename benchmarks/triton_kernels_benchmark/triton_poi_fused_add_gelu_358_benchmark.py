"""
Benchmark wrapper for the standalone Triton pointwise add+GELU kernels extracted
from Inductor output.

This family includes the variants:
  - triton_poi_fused_add_gelu_344
  - triton_poi_fused_add_gelu_346
  - triton_poi_fused_add_gelu_347 (add + GELU + tanh)
  - triton_poi_fused_add_gelu_352
  - triton_poi_fused_add_gelu_354
  - triton_poi_fused_add_gelu_355 (add + GELU + tanh)
  - triton_poi_fused_add_gelu_358
  - triton_poi_fused_add_gelu_360
  - triton_poi_fused_add_gelu_361 (add + GELU + tanh)

By default this benchmark loads the standalone kernel dump for
`triton_poi_fused_add_gelu_358`.
Use `kernel_name=` to select another family member, or override the kernel path
with `TRITON_POI_FUSED_ADD_GELU_PATH`.
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

DEFAULT_KERNEL_NAME = "triton_poi_fused_add_gelu_358"
LEGACY_KERNEL_PATH_ENV_VAR = "TRITON_POI_FUSED_ADD_GELU_358_PATH"
KERNEL_PATH_ENV_VAR = "TRITON_POI_FUSED_ADD_GELU_PATH"

KERNEL_VARIANTS = {
    "triton_poi_fused_add_gelu_344": {
        "path": "disassembles/inductor_test_log/af/cafx6kdholpbn2kcvxztf5ewvyu4sz7ufvwd6ek7ogoarnvj3ljn.py",
        "shape": (16, 1280, 80),
        "broadcast": (16, 1, 80),
        "op": "add_gelu",
    },
    "triton_poi_fused_add_gelu_346": {
        "path": "disassembles/inductor_test_log/5q/c5qemsbdzq2m4g5e5iqtrcagxcwoyj7ches5rtzazauygz4sjbi3.py",
        "shape": (16, 1280, 960),
        "broadcast": (16, 1, 960),
        "op": "add_gelu",
    },
    "triton_poi_fused_add_gelu_mul_tanh_347": {
        "path": "disassembles/inductor_test_log/tt/cttn54o7g2ptzekaj64ggjbi56mephmpagon5ggbcmko2ssiyazu.py",
        "shape": (16, 1280, 1280),
        "broadcast": (16, 1, 1280),
        "op": "add_gelu_mul_tanh",
    },
    "triton_poi_fused_add_gelu_352": {
        "path": "disassembles/inductor_test_log/mq/cmqgwithww3lpzv4uldcwaz6d62mb2h4ez7uba3ydb6mbcyeesxo.py",
        "shape": (16, 512, 200),
        "broadcast": (16, 1, 200),
        "op": "add_gelu",
    },
    "triton_poi_fused_add_gelu_354": {
        "path": "disassembles/inductor_test_log/2e/c2ewxbyafjtokby7wafdvdm77qhekhxtxlnkx2djiyo2zloekaeq.py",
        "shape": (16, 512, 960),
        "broadcast": (16, 1, 960),
        "op": "add_gelu",
    },
    "triton_poi_fused_add_gelu_mul_tanh_355": {
        "path": "disassembles/inductor_test_log/3b/c3brqzr7q67ksslutmajnh7k6g4ftf5wiohqugcwplekfw66t3l6.py",
        "shape": (16, 512, 1280),
        "broadcast": (16, 1, 1280),
        "op": "add_gelu_mul_tanh",
    },
    "triton_poi_fused_add_gelu_358": {
        "path": "disassembles/inductor_test_log/yp/cyptkx4fx7yvuj7o74ec6fopi4khkagiqdygza6cxqlr2yfpoql3.py",
        "shape": (16, 256, 160),
        "broadcast": (16, 1, 160),
        "op": "add_gelu",
    },
    "triton_poi_fused_add_gelu_360": {
        "path": "disassembles/inductor_test_log/pl/cplhanvuy5o4vwmqukdwa4b6777xfwgmc4piz3tweul23r6lqive.py",
        "shape": (16, 256, 960),
        "broadcast": (16, 1, 960),
        "op": "add_gelu",
    },
    "triton_poi_fused_add_gelu_mul_tanh_361": {
        "path": "disassembles/inductor_test_log/fp/cfpnooiq5tsxrbvfborp7msluhqufpvsbr53726ibm3cierhmzh6.py",
        "shape": (16, 256, 1280),
        "broadcast": (16, 1, 1280),
        "op": "add_gelu_mul_tanh",
    },
}

DEFAULT_SHAPES = [KERNEL_VARIANTS[DEFAULT_KERNEL_NAME]["shape"]]


def _resolve_kernel_name(kernel_name: Optional[str]) -> str:
    if kernel_name:
        return kernel_name
    env_name = os.getenv("TRITON_POI_FUSED_ADD_GELU_NAME")
    if env_name in KERNEL_VARIANTS:
        return env_name
    return DEFAULT_KERNEL_NAME


def _resolve_kernel_path(kernel_name: Optional[str], kernel_path: Optional[str]) -> Path:
    candidate = kernel_path or os.getenv(KERNEL_PATH_ENV_VAR) or os.getenv(LEGACY_KERNEL_PATH_ENV_VAR)
    if candidate:
        path = Path(candidate).expanduser()
    else:
        path = Path(__file__).resolve().parents[2] / KERNEL_VARIANTS[kernel_name or DEFAULT_KERNEL_NAME]["path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find extracted kernel file '{path}'. Set {KERNEL_PATH_ENV_VAR} or {LEGACY_KERNEL_PATH_ENV_VAR} to the standalone kernel path."
        )
    return path.resolve()


@functools.lru_cache(maxsize=None)
def _load_kernel(kernel_path: str, kernel_name: str):
    spec = importlib.util.spec_from_file_location(f"{kernel_name}_module", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create an import spec for '{kernel_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, kernel_name)
    except AttributeError as exc:
        raise AttributeError(f"'{kernel_path}' does not define '{kernel_name}'.") from exc


def _get_xpu_raw_stream():
    from torch._C import _xpu_getCurrentRawStream as get_raw_stream  # pylint: disable=C0415

    return get_raw_stream(torch.xpu.current_device())


def _make_inputs(shape: tuple[int, int, int], secondary_shape: tuple[int, int, int], op: str):
    batch, rows, cols = shape
    if rows <= 0 or cols <= 0 or batch <= 0:
        raise ValueError(f"Invalid input shape {shape} for add+GELU benchmark.")
    if batch != secondary_shape[0] or cols != secondary_shape[2]:
        raise ValueError(f"Expected shape {shape} to match the broadcast pattern {secondary_shape}.")

    torch.manual_seed(0)
    in_out = torch.randn((batch, rows, cols), device=DEVICE, dtype=torch.float16)
    in_ptr0 = torch.randn((batch, 1, cols), device=DEVICE, dtype=torch.float16)
    if op == "add_gelu_mul_tanh":
        in_ptr1 = torch.randn((batch, rows, cols), device=DEVICE, dtype=torch.float16)
        in_ptr2 = torch.randn((batch, 1, cols), device=DEVICE, dtype=torch.float16)
        return in_out, in_ptr0, in_ptr1, in_ptr2
    return in_out, in_ptr0


def _run_reference_add_gelu(in_out: torch.Tensor, in_ptr0: torch.Tensor) -> torch.Tensor:
    return F.gelu(in_out.to(torch.float32) + in_ptr0.to(torch.float32), approximate="none").to(torch.float16)


def _run_reference_add_gelu_mul_tanh(
    in_out: torch.Tensor,
    in_ptr0: torch.Tensor,
    in_ptr1: torch.Tensor,
    in_ptr2: torch.Tensor,
) -> torch.Tensor:
    base = F.gelu(in_out.to(torch.float32) + in_ptr0.to(torch.float32), approximate="none")
    tanh_term = torch.tanh((in_ptr1 + in_ptr2).to(torch.float32))
    return (base * tanh_term).to(torch.float16)


def _run_triton(kernel, in_out: torch.Tensor, in_out_template: torch.Tensor, inputs: tuple[torch.Tensor, ...],
                xnumel: int, op: str):
    if DEVICE != "xpu":
        raise RuntimeError(f"Triton add+GELU kernels were extracted for XPU and cannot run on '{DEVICE}'.")
    in_out.copy_(in_out_template)
    flat = [t.reshape(-1) for t in (in_out, *inputs)]
    if op == "add_gelu_mul_tanh":
        kernel.run(flat[0], flat[1], flat[2], flat[3], xnumel, stream=_get_xpu_raw_stream())
    else:
        kernel.run(flat[0], flat[1], xnumel, stream=_get_xpu_raw_stream())
    return in_out


def _perf_rate(work: float, ms: float) -> float:
    if ms == 0:
        return float("inf")
    return work / (ms * 1e-3)


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    kernel_name: Optional[str] = None,
    kernel_path: Optional[str] = None,
    x_vals: Optional[list[tuple[int, int, int]]] = None,
):
    """Build a benchmark report for a selected add+GELU Triton kernel family member."""

    resolved_kernel_name = _resolve_kernel_name(kernel_name)
    variant = KERNEL_VARIANTS[resolved_kernel_name]
    supported_providers = {"torch": "Torch"}
    if DEVICE == "xpu":
        supported_providers = {"triton": "Triton", **supported_providers}
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    resolved_kernel_path = _resolve_kernel_path(resolved_kernel_name, kernel_path) if "triton" in providers else None
    x_vals = x_vals if x_vals is not None else [variant["shape"]]

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["batch", "rows", "cols"],
            x_vals=[shape for shape in x_vals],
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("green", "-"), ("blue", "--")],
            ylabel=["GB/s", "TFlops"],
            plot_name=f"triton-poi-fused-add-gelu-{resolved_kernel_name.rsplit('_', 1)[-1]}",
            args={},
        ))
    def benchmark(batch, rows, cols, provider):
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        inputs = _make_inputs((batch, rows, cols), variant["broadcast"], variant["op"])
        in_out_template = inputs[0]
        other_inputs = inputs[1:]

        if provider == "triton":
            if resolved_kernel_path is None:
                raise RuntimeError(f"{resolved_kernel_name} requires a standalone kernel path.")
            kernel = _load_kernel(str(resolved_kernel_path), resolved_kernel_name)
            triton_out = in_out_template.clone()
            if variant["op"] == "add_gelu_mul_tanh":
                ref_out = _run_reference_add_gelu_mul_tanh(*inputs)
                triton_fn = lambda: _run_triton(kernel, triton_out, in_out_template, other_inputs, batch * rows * cols,
                                                variant["op"])
                torch_fn = lambda: ref_out
            else:
                ref_out = _run_reference_add_gelu(*inputs[:2])
                triton_fn = lambda: _run_triton(kernel, triton_out, in_out_template, other_inputs, batch * rows * cols,
                                                variant["op"])
                torch_fn = lambda: ref_out
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-3, rtol=1e-3, err_msg=resolved_kernel_name)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
        elif provider == "torch":
            if variant["op"] == "add_gelu_mul_tanh":
                torch_fn = lambda: _run_reference_add_gelu_mul_tanh(*inputs)
            else:
                torch_fn = lambda: _run_reference_add_gelu(*inputs[:2])
            _, min_ms, max_ms, mean_ms, cv = do_bench(torch_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        total_bytes = batch * rows * cols * (2 + 2 + 2)
        total_flops = batch * rows * cols * 6
        if variant["op"] == "add_gelu_mul_tanh":
            total_bytes += batch * rows * cols * 2 * 2
            total_flops += batch * rows * cols * 4
        gbps = lambda ms: _perf_rate(total_bytes * 1e-9, ms)
        tflops = lambda ms: _perf_rate(total_flops * 1e-12, ms)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
