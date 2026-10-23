"""Benchmark the unfused GEMM shapes reported by the fx repros.

This benchmark loads the bad-shape list from
``disassembles/fx_repros/unfused_gemm_shapes.json`` and benchmarks the
corresponding Triton GEMM kernels against the same logical shapes in PyTorch.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark.benchmark_testing import DEVICE

# Resolve the benchmark data relative to the repo root by default, but allow
# callers to override it via an environment variable when running from a copied
# or otherwise non-standard checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_JSON_PATH = REPO_ROOT / "disassembles" / "fx_repros" / "unfused_gemm_shapes.json"
_JSON_PATH = Path(os.environ.get("TRITON_UNFUSED_GEMM_JSON", _DEFAULT_JSON_PATH))
_DEFAULT_SHAPE_LIMIT = 10
_SHAPE_LIMIT = int(os.environ.get("TRITON_UNFUSED_GEMM_SHAPE_LIMIT", _DEFAULT_SHAPE_LIMIT))


def _load_entries():
    with _JSON_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["unfused_gemms"]


def _slowest_entries(entries, limit: int = _SHAPE_LIMIT):
    ranked = sorted(
        entries,
        key=lambda entry: float(entry.get("ratio_triton_over_onednn", 0.0)),
        reverse=True,
    )
    if limit <= 0:
        return []
    return ranked[:limit]


# Use only the slowest entries so the benchmark stays focused on the worst
# Triton regressions from the collected repro list.
def _selected_entries():
    return _slowest_entries(_load_entries(), _SHAPE_LIMIT)


def _shape_key(entry) -> Tuple[int, int, int, int]:
    a = entry["matmul"]["a"]
    b = entry["matmul"]["b"]
    if len(a) == 2 and len(b) == 2:
        return (1, a[0], b[1], a[1])
    if len(a) == 3 and len(b) == 3:
        return (a[0], a[1], b[2], a[2])
    raise ValueError(f"Unsupported GEMM entry: {entry}")


def _lookup_entry(B: int, M: int, N: int, K: int):
    for entry in _selected_entries():
        if _shape_key(entry) == (B, M, N, K):
            return entry
    raise KeyError(f"No entry for B={B}, M={M}, N={N}, K={K}")


def _torch_reference(op: str, a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor] = None):
    if op == "mm":
        return a @ b
    if op == "bmm":
        return torch.bmm(a, b)
    if op == "addmm":
        if bias is None:
            raise ValueError("bias is required for addmm")
        return torch.addmm(bias, a, b)
    raise ValueError(f"Unsupported op: {op}")


def _compiled_triton_reference(op: str, a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor] = None):
    import torch._inductor.config as cfg

    cfg.max_autotune_gemm = True
    cfg.max_autotune_gemm_backends = "TRITON"

    if op == "mm":
        fn = lambda x, y: x @ y
        compiled = torch.compile(fn, backend="inductor", mode="max-autotune-no-cudagraphs")
        return compiled(a, b)
    if op == "bmm":
        fn = lambda x, y: torch.bmm(x, y)
        compiled = torch.compile(fn, backend="inductor", mode="max-autotune-no-cudagraphs")
        return compiled(a, b)
    if op == "addmm":
        if bias is None:
            raise ValueError("bias is required for addmm")
        fn = lambda x, y, z: torch.addmm(z, x, y)
        compiled = torch.compile(fn, backend="inductor", mode="max-autotune-no-cudagraphs")
        return compiled(a, b, bias)
    raise ValueError(f"Unsupported op: {op}")


def _make_input(entry, device: str):
    a_shape = entry["matmul"]["a"]
    b_shape = entry["matmul"]["b"]
    op = entry["op"]

    if len(a_shape) == 2 and len(b_shape) == 2:
        a = torch.rand((a_shape[0], a_shape[1]), device=device, dtype=torch.bfloat16)
        b = torch.rand((b_shape[0], b_shape[1]), device=device, dtype=torch.bfloat16)
        bias = torch.rand((a_shape[0], b_shape[1]), device=device, dtype=torch.bfloat16) if op == "addmm" else None
        return op, a, b, bias

    if len(a_shape) == 3 and len(b_shape) == 3:
        a = torch.rand(tuple(a_shape), device=device, dtype=torch.bfloat16)
        b = torch.rand(tuple(b_shape), device=device, dtype=torch.bfloat16)
        bias = None
        return op, a, b, bias

    raise ValueError(f"Unsupported GEMM entry: {entry}")


def _flop_count(op: str, B: int, M: int, N: int, K: int) -> float:
    return 2.0 * B * M * N * K


def get_benchmark(
    providers_filter: Optional[List[str]] = None,
    plot_name: str = "unfused-gemm-performance",
):
    supported_providers = {
        "triton": "Triton",
        "onednn": "OneDNN",
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    x_vals = []
    for entry in _selected_entries():
        x_vals.append(_shape_key(entry))

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["B", "M", "N", "K"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("green", "-"), ("green", "--"), ("blue", "-")],
            ylabel=["GB/s", "TFlops"],
            plot_name=plot_name,
            args={},
        ))
    def benchmark(B, M, N, K, provider):
        entry = _lookup_entry(B, M, N, K)
        op = entry["op"]

        torch.manual_seed(0)
        op_name, a, b, bias = _make_input(entry, DEVICE)
        assert op_name == op

        if provider == "onednn":
            if op == "mm":
                torch_fn = lambda: a @ b
            elif op == "bmm":
                torch_fn = lambda: torch.bmm(a, b)
            elif op == "addmm":
                assert bias is not None
                torch_fn = lambda: torch.addmm(bias.to(torch.float32), a.to(torch.float32), b.to(torch.float32))
            else:
                raise NotImplementedError(f"Unsupported op {op}")
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.get_do_bench(
                n_warmup=800,
                n_repeat=10,
                quantiles=[0.5, 0.0, 1.0],
            )(torch_fn)

        elif provider == "triton":
            torch_fn = lambda: _torch_reference(op, a, b, bias)
            triton_fn = lambda: _compiled_triton_reference(op, a, b, bias)
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-4, rtol=1e-3, err_msg=f"{op} to torch")
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.get_do_bench(
                n_warmup=800,
                n_repeat=10,
                quantiles=[0.5, 0.0, 1.0],
            )(triton_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        flops = _flop_count(op, B, M, N, K)
        tflops = lambda ms: flops * 1e-12 / (ms * 1e-3)
        gbps = lambda ms: B * (2 * (M * K + K * N) + 4.0 * (M * N)) * 1e-9 / (ms * 1e-3)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    benchmark = get_benchmark()
    benchmark.run(show_plots=False, print_data=True)
